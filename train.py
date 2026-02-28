import torch
from torch import nn
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
from models.utils import print_model_grads


class PlumTrainer:
    def __init__(self, config):
        """初始化训练器"""
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.save_dir = "./checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)

        self._setup()

        self.early_stop_patience = 15  # 连续无提升的epoch阈值
        self.early_stop_counter = 0  # 连续无提升计数器
        self.improvement_threshold = 1e-4  # 显著提升的阈值（0.01%）
        self.best_val_metrics = {  # 保存最佳验证集所有指标
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
            num_workers=6
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.config['batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=2
        )
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.config['batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=2
        )

        self.model = createPeftPlumModel(
            self.config['modes'],
            self.config['classes_num']
        ).to(self.device)

        total_samples = sum(self.config['class_counts'])
        class_weights = torch.tensor(
            [total_samples / count for count in self.config['class_counts']],
            dtype=torch.float32
        ).to(self.device)

        self.criterion_1 = WeightedFocalLoss(class_weights=class_weights).to(self.device)
        self.criterion_2 = WeightedFocalLoss(class_weights=class_weights).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['lr'],
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=True
        )

        self.best_val_acc = 0.0
        self.best_model_path = os.path.join(self.save_dir, "best_model.pth")

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
            'test_map': []
        }

    def _calculate_metrics(self, y_true, y_pred, y_score):

        num_classes = self.config['classes_num']


        acc = 100.0 * accuracy_score(y_true, y_pred)


        try:
            if num_classes == 2:
                auc = 100.0 * roc_auc_score(y_true, y_score[:, 1])
            else:
                y_true_binarized = label_binarize(y_true, classes=range(num_classes))
                auc = 100.0 * roc_auc_score(y_true_binarized, y_score, multi_class='ovr')
        except Exception as e:
            print(f"计算AUC时出错: {e}，默认设为0.0")
            auc = 0.0


        f1 = 100.0 * f1_score(y_true, y_pred, average='macro', zero_division=0)


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
        print(f"模型已保存到: {path}")

    def load_model(self, path):

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_metrics = checkpoint.get('best_val_metrics', {'acc': 0.0, 'auc': 0.0, 'f1': 0.0, 'map': 0.0})
        print(f"模型已从 {path} 加载")

    def train(self):

        print(f"\n{'=' * 80}")
        print("开始训练")
        print(f"{'=' * 80}\n")

        for epoch in range(self.config['epochs']):
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            print('=' * 80)

            train_loss, train_acc = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            print(f"\n训练集 - 平均损失: {train_loss:.4f}, 准确度: {train_acc:.2f}%")


            val_loss, val_acc, val_auc, val_f1, val_map, val_class_correct, val_class_total = self.validate(
                self.val_loader, 'val')


            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['val_f1'].append(val_f1)
            self.history['val_map'].append(val_map)


            print(f"验证集 - 平均损失: {val_loss:.4f}")
            print(f"验证集核心指标：")
            print(f"  准确率(ACC): {val_acc:.2f}%")
            print(f"  AUC: {val_auc:.2f}%")
            print(f"  F1-Score: {val_f1:.2f}%")
            print(f"  mAP: {val_map:.2f}%")


            print("各类别验证准确度:")
            for i in range(self.config['classes_num']):
                if val_class_total[i] > 0:
                    class_acc = 100.0 * val_class_correct[i] / val_class_total[i]
                    print(f"  类别 {i}: {class_acc:.2f}% ({val_class_correct[i]}/{val_class_total[i]})")
                else:
                    print(f"  类别 {i}: 无样本")


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
                print(f"新的最佳验证指标，模型已保存！")
                print(f"  最佳ACC: {self.best_val_metrics['acc']:.2f}%")
                print(f"  最佳AUC: {self.best_val_metrics['auc']:.2f}%")
                print(f"  最佳F1: {self.best_val_metrics['f1']:.2f}%")
                print(f"  最佳mAP: {self.best_val_metrics['map']:.2f}%")
            else:

                self.early_stop_counter += 1
                print(f"早停计数器: {self.early_stop_counter}/{self.early_stop_patience}")


                if self.early_stop_counter >= self.early_stop_patience:
                    print(f"\n验证集指标连续{self.early_stop_patience}个epoch无显著提升，触发早停！")
                    print(f"最佳验证ACC: {self.best_val_metrics['acc']:.2f}%")
                    break

            print("-" * 80)

        print("\n" + "=" * 80)
        print("训练完成!")
        print("=" * 80)

    def test(self):

        print("\n[使用最佳模型进行测试]")


        self.load_model(self.best_model_path)


        test_loss, test_acc, test_auc, test_f1, test_map, test_class_correct, test_class_total = self.validate(
            self.test_loader, 'test')

        self.history['test_loss'].append(test_loss)
        self.history['test_acc'].append(test_acc)
        self.history['test_auc'].append(test_auc)
        self.history['test_f1'].append(test_f1)
        self.history['test_map'].append(test_map)


        print(f"\n测试集 - 平均损失: {test_loss:.4f}")
        print(f"测试集核心指标：")
        print(f"  准确率(ACC): {test_acc:.2f}%")
        print(f"  AUC: {test_auc:.2f}%")
        print(f"  F1-Score: {test_f1:.2f}%")
        print(f"  mAP: {test_map:.2f}%")


        print("\n各类别测试准确度:")
        for i in range(self.config['classes_num']):
            if test_class_total[i] > 0:
                class_acc = 100.0 * test_class_correct[i] / test_class_total[i]
                print(f"  类别 {i}: {class_acc:.2f}% ({test_class_correct[i]}/{test_class_total[i]})")
            else:
                print(f"  类别 {i}: 无样本")


        if any(test_class_total):
            valid_classes = [i for i in range(self.config['classes_num']) if test_class_total[i] > 0]
            macro_avg = 100.0 * sum([test_class_correct[i] / test_class_total[i] for i in valid_classes]) / len(
                valid_classes)
            micro_avg = test_acc
            print(f"\n宏平均准确度: {macro_avg:.2f}%")
            print(f"微平均准确度: {micro_avg:.2f}%")


        print("\n" + "=" * 80)
        print("最终结果汇总:")
        print("=" * 80)
        print(f"最佳验证ACC: {self.best_val_metrics['acc']:.2f}%")
        print(f"最佳验证AUC: {self.best_val_metrics['auc']:.2f}%")
        print(f"最佳验证F1: {self.best_val_metrics['f1']:.2f}%")
        print(f"最佳验证mAP: {self.best_val_metrics['map']:.2f}%")
        print(f"测试集ACC: {test_acc:.2f}%")
        print(f"测试集AUC: {test_auc:.2f}%")
        print(f"测试集F1: {test_f1:.2f}%")
        print(f"测试集mAP: {test_map:.2f}%")

        return self.history


def main():
    """主函数"""

    config = {
        'input_size': 256,
        'batch_size': 128,
        'class_counts': [474, 1152, 954, 492],
        'classes_num': 4,
        'lr': 0.001,
        'alpha': 0.1,
        'epochs': 200,
        'modes': ['weight', "sugar","persistent"]
    }


    trainer = PlumTrainer(config)
    trainer.train()
    history = trainer.test()

    return history


if __name__ == "__main__":
    main()