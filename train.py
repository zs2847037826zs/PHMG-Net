import argparse
import torch
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

from dataset.plum_dataset import PlumDataset
from models.plum_swin_transformer import createPeftPlumModel, WeightedFocalLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class PlumTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_dir = "./checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)
        self._setup()
        self.early_stop_patience = self.config.get('early_stop_patience', 15)
        self.early_stop_counter = 0
        self.improvement_threshold = self.config.get('improvement_threshold', 1e-4)
        self.best_val_metrics = {
            'acc': 0.0,
            'auc': 0.0,
            'f1': 0.0,
            'map': 0.0
        }

    def _setup(self):
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['input_size'], self.config['input_size']), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.01),
            transforms.RandomRotation(180),
            transforms.RandomAffine(180, translate=[0.1, 0.1], scale=[0.7, 1.3]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.vt_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config['input_size'], self.config['input_size']), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_set = PlumDataset("./data", transform=self.train_transform)
        self.val_set = PlumDataset("./data", transform=self.vt_transform, mode='val')
        self.test_set = PlumDataset("./data", transform=self.vt_transform, mode='test')

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=1,
            persistent_workers=True
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.config['batch_size'],
            shuffle=False,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.config['batch_size'],
            shuffle=False,
            pin_memory=True,
        )

        # 模型
        self.model = createPeftPlumModel(
            self.config['modes'],
            self.config['classes_num']
        ).to(self.device)

        # 类别权重（可选）
        total_samples = sum(self.config['class_counts'])
        class_weights = torch.tensor(
            [total_samples / count for count in self.config['class_counts']],
            dtype=torch.float32
        ).to(self.device)

        # 损失函数
        self.criterion_1 = WeightedFocalLoss(class_weights=class_weights).to(self.device)
        self.criterion_2 = WeightedFocalLoss(class_weights=class_weights).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['lr'],
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=True
        )

        # 记录最佳模型
        self.best_val_acc = 0.0
        self.best_model_path = os.path.join(
            self.save_dir,
            f"best_model_{''.join(item[0] for item in self.config['modes'])}.pth"
        )

        # 历史记录（扩展：加入所有评估指标）
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'val_f1': [],
            'val_map': [],
            'test_loss': [],
            'test_acc': [],
            'test_auc': [],
            'test_f1': [],
            'test_map': [],
            'train_acc': []
        }

    def _calculate_metrics(self, y_true, y_pred, y_score):
        """
        封装指标计算逻辑（适配sklearn 1.2.0）
        """
        num_classes = self.config['classes_num']

        # 1. 准确率 (ACC)
        acc = 100.0 * accuracy_score(y_true, y_pred)

        # 2. AUC（多分类：One-vs-Rest）
        try:
            if num_classes == 2:
                auc = 100.0 * roc_auc_score(y_true, y_score[:, 1])
            else:
                y_true_binarized = label_binarize(y_true, classes=range(num_classes))
                auc = 100.0 * roc_auc_score(y_true_binarized, y_score, multi_class='ovr')
        except Exception as e:
            print(f"计算AUC时出错: {e}，默认设为0.0")
            auc = 0.0

        # 3. F1值（宏平均）
        f1 = 100.0 * f1_score(y_true, y_pred, average='macro', zero_division=0)

        # 4. mAP
        try:
            if num_classes == 2:
                ap = average_precision_score(y_true, y_score[:, 1])
                map_score = 100.0 * ap
            else:
                y_true_binarized = label_binarize(y_true, classes=range(num_classes))
                ap_per_class = []
                for i in range(num_classes):
                    try:
                        ap = average_precision_score(y_true_binarized[:, i], y_score[:, i])
                    except ZeroDivisionError:
                        ap = 0.0
                    ap_per_class.append(ap)
                map_score = 100.0 * np.mean(ap_per_class)
        except Exception as e:
            print(f"计算mAP时出错: {e}，默认设为0.0")
            map_score = 0.0

        return acc, auc, f1, map_score

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(
            self.train_loader,
            desc=f'Training Epoch {epoch + 1}/{self.config["epochs"]}',
            unit='batch',
            leave=True,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )

        for batch_idx, (data, target, pd, weight, sugar, path) in enumerate(train_bar):
            data = data.float().to(self.device)
            target = target.long().to(self.device)
            pd = pd.float().to(self.device if 'persistent' in self.config['modes'] else torch.device('cpu'))
            weight = weight.float().to(self.device)
            sugar = sugar.float().to(self.device)

            predict, topo = self.model([data, pd, weight, sugar])

            loss1 = self.criterion_1(predict, target)
            if topo is not None:
                loss2 = self.criterion_2(topo, target)
                loss = loss1 + self.config['alpha'] * loss2
            else:
                loss = loss1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(predict, 1)
            train_correct += (predicted == target).sum().item()
            train_total += target.size(0)

            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * train_correct / train_total:.1f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * train_correct / train_total

        return avg_loss, accuracy

    def validate(self, data_loader, mode='val'):
        """验证/测试"""
        self.model.eval()
        total_loss = 0
        y_true = []
        y_pred = []
        y_score = []

        class_correct = [0] * self.config['classes_num']
        class_total = [0] * self.config['classes_num']

        with torch.no_grad():
            val_bar = tqdm(
                data_loader,
                desc=f'{mode.capitalize()}',
                unit='batch',
                leave=True,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            )

            for batch_idx, (data, target, pd, weight, sugar, path) in enumerate(val_bar):
                data = data.float().to(self.device)
                target = target.long().to(self.device)
                pd = pd.float().to(self.device if 'persistent' in self.config['modes'] else torch.device('cpu'))
                weight = weight.float().to(self.device)
                sugar = sugar.float().to(self.device)

                predict, topo = self.model([data, pd, weight, sugar])

                loss1 = self.criterion_1(predict, target)
                if 'persistent' in self.config['modes'] and topo is not None:
                    loss2 = self.criterion_2(topo, target)
                    loss = loss1 + self.config['alpha'] * loss2
                else:
                    loss = loss1
                total_loss += loss.item()

                prob = torch.softmax(predict, dim=1)
                _, predicted = torch.max(predict, 1)

                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_score.extend(prob.cpu().numpy())

                for i in range(target.size(0)):
                    label = target[i].item()
                    if 0 <= label < self.config['classes_num']:
                        class_total[label] += 1
                        if predicted[i].item() == label:
                            class_correct[label] += 1

                current_acc = 100.0 * sum(class_correct) / max(1, sum(class_total))
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.1f}%'
                })

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)

        acc, auc, f1, map_score = self._calculate_metrics(y_true, y_pred, y_score)
        avg_loss = total_loss / len(data_loader)

        return avg_loss, acc, auc, f1, map_score, class_correct, class_total

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_val_metrics': self.best_val_metrics
        }, path)
        print(f"Model saved to: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_metrics = checkpoint.get('best_val_metrics', {'acc': 0.0, 'auc': 0.0, 'f1': 0.0, 'map': 0.0})
        print(f"Model loaded from {path}")

    def train(self):
        print(f"\n{'=' * 80}")
        print("Start training")
        print(f"{'=' * 80}\n")

        for epoch in range(self.config['epochs']):
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            print('=' * 80)

            train_loss, train_acc = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            print(f"\nTrain set - Average Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

            val_loss, val_acc, val_auc, val_f1, val_map, val_class_correct, val_class_total = self.validate(
                self.val_loader, 'val')

            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['val_f1'].append(val_f1)
            self.history['val_map'].append(val_map)

            print(f"Validation set - Average Loss: {val_loss:.4f}")
            print("Validation metrics:")
            print(f"  Accuracy (ACC): {val_acc:.2f}%")
            print(f"  AUC: {val_auc:.2f}%")
            print(f"  F1-Score: {val_f1:.2f}%")
            print(f"  mAP: {val_map:.2f}%")

            print("Per-class validation accuracy:")
            for i in range(self.config['classes_num']):
                if val_class_total[i] > 0:
                    class_acc = 100.0 * val_class_correct[i] / val_class_total[i]
                    print(f"  Class {i}: {class_acc:.2f}% ({val_class_correct[i]}/{val_class_total[i]})")
                else:
                    print(f"  Class {i}: No samples")

            if val_acc > self.best_val_metrics['acc'] + self.improvement_threshold:
                self.best_val_metrics = {
                    'acc': val_acc,
                    'auc': val_auc,
                    'f1': val_f1,
                    'map': val_map
                }
                self.best_val_acc = val_acc
                self.save_model(self.best_model_path)
                self.early_stop_counter = 0
                print(f"  ACC: {self.best_val_metrics['acc']:.2f}%")
                print(f"  AUC: {self.best_val_metrics['auc']:.2f}%")
                print(f"  F1: {self.best_val_metrics['f1']:.2f}%")
                print(f"  mAP: {self.best_val_metrics['map']:.2f}%")
            else:
                self.early_stop_counter += 1
                print(f"Early stop counter: {self.early_stop_counter}/{self.early_stop_patience}")

                if self.early_stop_counter >= self.early_stop_patience:
                    print(f"\nNo significant improvement in validation metrics for {self.early_stop_patience} consecutive epochs. Early stopping triggered!")
                    print(f"Best validation ACC: {self.best_val_metrics['acc']:.2f}%")
                    break

            print("-" * 80)

        print("\n" + "=" * 80)
        print("Training completed!")
        print("=" * 80)

    def test(self):
        self.load_model(self.best_model_path)
        test_loss, test_acc, test_auc, test_f1, test_map, test_class_correct, test_class_total = self.validate(
            self.test_loader, 'test')

        self.history['test_loss'].append(test_loss)
        self.history['test_acc'].append(test_acc)
        self.history['test_auc'].append(test_auc)
        self.history['test_f1'].append(test_f1)
        self.history['test_map'].append(test_map)

        print("\nPer-class test accuracy:")
        for i in range(self.config['classes_num']):
            if test_class_total[i] > 0:
                class_acc = 100.0 * test_class_correct[i] / test_class_total[i]
                print(f"  Class {i}: {class_acc:.2f}% ({test_class_correct[i]}/{test_class_total[i]})")
            else:
                print(f"  Class {i}: No samples")

        if any(test_class_total):
            valid_classes = [i for i in range(self.config['classes_num']) if test_class_total[i] > 0]
            macro_avg = 100.0 * sum([test_class_correct[i] / test_class_total[i] for i in valid_classes]) / len(
                valid_classes)
            micro_avg = test_acc
            print(f"\nMacro-average accuracy: {macro_avg:.2f}%")
            print(f"Micro-average accuracy: {micro_avg:.2f}%")

        print("\n" + "=" * 80)
        print("Summary of final results:")
        print("=" * 80)
        print(f"Best validation ACC: {self.best_val_metrics['acc']:.2f}%")
        print(f"Best validation AUC: {self.best_val_metrics['auc']:.2f}%")
        print(f"Best validation F1: {self.best_val_metrics['f1']:.2f}%")
        print(f"Best validation mAP: {self.best_val_metrics['map']:.2f}%")
        print(f"Test set ACC: {test_acc:.2f}%")
        print(f"Test set AUC: {test_auc:.2f}%")
        print(f"Test set F1: {test_f1:.2f}%")
        print(f"Test set mAP: {test_map:.2f}%")

        return self.history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plum Swin Transformer 训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Basic configurations
    parser.add_argument('--input_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--classes_num', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Weight for the second loss term')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Total training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed')

    # Number of samples per class: separated by commas, e.g., --class_counts 474,1152,954,492
    parser.add_argument('--class_counts', type=str, default='474,1152,954,492',
                        help='Number of samples for each class (comma-separated)')

    # Modalities to use: separated by spaces, e.g., --modes weight sugar persistent
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['weight', 'sugar', 'persistent'],
                        help='Modalities to use, multiple options available: weight sugar persistent')

    # Early stopping settings
    parser.add_argument('--early_stop_patience', type=int, default=15,
                        help='Patience for early stopping (number of consecutive epochs without improvement)')
    parser.add_argument('--improvement_threshold', type=float, default=1e-4,
                        help='Threshold for judging a significant improvement')
    args = parser.parse_args()

    if isinstance(args.class_counts, str):
        args.class_counts = [int(x) for x in args.class_counts.split(',') if x.strip() != '']

    config = vars(args)

    if len(config['class_counts']) != config['classes_num']:
        raise ValueError(
            f"The length of class_counts ({len(config['class_counts'])}) "
            f"must match classes_num ({config['classes_num']})"
        )

    return config

def main():
    config = parse_args()
    set_seed(config['seed'])
    print("=" * 80)
    print("Training configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    trainer = PlumTrainer(config)
    trainer.train()
    history = trainer.test()

    return history


if __name__ == "__main__":
    main()
