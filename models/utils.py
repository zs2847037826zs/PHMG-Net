from typing import Dict, Tuple

import torch
from peft.utils import ModulesToSaveWrapper
from torch import nn


def print_model_params(model):
    total_params = 0
    trainable_params = 0

    print("=" * 60)
    print("模型参数统计")
    print("=" * 60)

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

        print(f"{name:40s} | 形状: {str(param.shape):20s} | "
              f"参数量: {num_params:8,} | "
              f"{'可训练' if param.requires_grad else '冻结'}")

    print("-" * 60)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    print(f"可训练参数占比: {trainable_params / total_params * 100:.2f}%")
    print("=" * 60)


def print_model_grads(model, loss=None):
    if loss is not None:
        model.zero_grad()
        loss.backward()

    print("=" * 60)
    print("模型梯度信息")
    print("=" * 60)

    has_grad_count = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += 1

        if param.grad is None:
            grad_status = "无梯度"
        else:
            has_grad_count += 1
            grad_norm = param.grad.norm().item()
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()

            if has_nan or has_inf:
                grad_status = f"范数: {grad_norm:.2e} ⚠️NaN/Inf"
            else:
                grad_status = f"范数: {grad_norm:.2e}"

        print(f"{name:40s} | {grad_status}")

    print("-" * 60)
    print(f"有梯度的参数: {has_grad_count}/{total_params}")

    # 检查梯度问题
    if has_grad_count > 0:
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        if grad_norms:
            avg_norm = sum(grad_norms) / len(grad_norms)
            max_norm = max(grad_norms)
            print(f"平均梯度范数: {avg_norm:.2e}")
            print(f"最大梯度范数: {max_norm:.2e}")

            if avg_norm < 1e-7:
                print("⚠️ 警告: 梯度可能消失")
            if max_norm > 1e3:
                print("⚠️ 警告: 梯度可能爆炸")

    print("=" * 60)


def copy_weights_to_modules_to_save(peft_model, original_model, modules_to_save):
    peft_base_model = peft_model.base_model
    original_base_model = original_model

    print("Copying weights from original model to PEFT model's trainable modules...")
    for module_name in modules_to_save:
        try:
            original_module = original_base_model
            for name_part in module_name:
                original_module = getattr(original_module, name_part)

            peft_module = peft_base_model
            for name_part in module_name.split('.'):
                peft_module = getattr(peft_module, name_part)

            peft_module.load_state_dict(original_module.state_dict())

            print(f"  - Successfully copied weights for module: {module_name}")

            with torch.no_grad():
                if not torch.equal(peft_module.weight, original_module.weight):
                    print(f"  ! Warning: Weights for {module_name} do not match after copy!")
                if hasattr(peft_module, 'bias') and peft_module.bias is not None:
                    if not torch.equal(peft_module.bias, original_module.bias):
                        print(f"  ! Warning: Bias for {module_name} do not match after copy!")

        except AttributeError:
            print(f"  - Error: Module '{module_name}' not found in either the original or PEFT model.")

    print("Weight copying complete.\n")


def compare_all_modules_to_save(peft_model):
    print("=== 开始对比所有ModulesToSaveWrapper的参数 ===")
    for module_name, wrapper_module in peft_model.named_modules():
        if isinstance(wrapper_module, ModulesToSaveWrapper):
            print(f"\n模块名: {module_name}")
            orig_module = wrapper_module.original_module
            saved_module = wrapper_module.modules_to_save["default"]

            all_params_match = True
            for (orig_param, saved_param) in zip(orig_module.parameters(), saved_module.parameters()):
                if not torch.allclose(orig_param, saved_param, atol=1e-8):
                    all_params_match = False
                    break

            # 输出结果
            if all_params_match:
                print("✅ 参数完全一致")
            else:
                print("❌ 参数不一致")
    print("\n=== 对比完成 ===")

def compare_model_parameters(
        model1: nn.Module,
        model2: nn.Module,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        verbose: bool = True
) -> Dict[str, Dict[str, any]]:

    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    # 找出两个模型共有的参数名
    common_names = set(params1.keys()) & set(params2.keys())
    only_in_model1 = set(params1.keys()) - set(params2.keys())
    only_in_model2 = set(params2.keys()) - set(params1.keys())

    results = {
        'common_parameters': {},
        'summary': {
            'total_common': len(common_names),
            'identical_count': 0,
            'different_count': 0,
            'only_in_model1': list(only_in_model1),
            'only_in_model2': list(only_in_model2),
            'only_in_model1_count': len(only_in_model1),
            'only_in_model2_count': len(only_in_model2)
        }
    }

    if verbose:
        print(f"模型1参数总数: {len(params1)}")
        print(f"模型2参数总数: {len(params2)}")
        print(f"共有参数数量: {len(common_names)}")
        print(f"仅在模型1中的参数数量: {len(only_in_model1)}")
        print(f"仅在模型2中的参数数量: {len(only_in_model2)}")

        if only_in_model1:
            print(f"\n仅在模型1中的参数: {list(only_in_model1)}")
        if only_in_model2:
            print(f"仅在模型2中的参数: {list(only_in_model2)}")
        print("-" * 80)

    # 比较共有参数
    for name in sorted(common_names):
        param1 = params1[name]
        param2 = params2[name]

        shape_match = param1.shape == param2.shape

        if shape_match:
            if torch.allclose(param1, param2, rtol=rtol, atol=atol):
                value_match = True
                diff_norm = torch.norm(param1 - param2).item()
                results['summary']['identical_count'] += 1
                status = "相同"
            else:
                value_match = False
                diff_norm = torch.norm(param1 - param2).item()
                results['summary']['different_count'] += 1
                status = "不同"
        else:
            value_match = False
            diff_norm = None
            results['summary']['different_count'] += 1
            status = "形状不同"

        # 存储比较结果
        results['common_parameters'][name] = {
            'shape1': tuple(param1.shape),
            'shape2': tuple(param2.shape),
            'shape_match': shape_match,
            'value_match': value_match,
            'dtype1': param1.dtype,
            'dtype2': param2.dtype,
            'norm1': torch.norm(param1).item() if shape_match else None,
            'norm2': torch.norm(param2).item() if shape_match else None,
            'diff_norm': diff_norm,
            'status': status
        }

        if verbose:
            print(f"参数: {name}")
            print(f"  形状: {tuple(param1.shape)} vs {tuple(param2.shape)} - {'相同' if shape_match else '不同'}")
            if shape_match:
                print(f"  范数: {torch.norm(param1).item():.6f} vs {torch.norm(param2).item():.6f}")
                print(f"  差异范数: {diff_norm:.6e}")
                print(f"  状态: {status}")
            else:
                print(f"  状态: {status}")
            print()

    # 打印总结
    if verbose:
        print("=" * 80)
        print("比较总结:")
        print(f"共有参数总数: {results['summary']['total_common']}")
        print(f"相同参数数量: {results['summary']['identical_count']}")
        print(f"不同参数数量: {results['summary']['different_count']}")

        if results['summary']['identical_count'] == results['summary']['total_common']:
            print("\n✅ 所有共有参数都相同！")
        elif results['summary']['different_count'] > 0:
            print(f"\n❌ 有 {results['summary']['different_count']} 个参数不同")
            print("不同的参数:")
            for name, info in results['common_parameters'].items():
                if not info['value_match']:
                    print(f"  - {name}: {info['status']}")

    return results


def check_models_identical(
        model1: nn.Module,
        model2: nn.Module,
        rtol: float = 1e-5,
        atol: float = 1e-8
) -> Tuple[bool, Dict[str, any]]:

    results = compare_model_parameters(model1, model2, rtol, atol, verbose=False)

    if (results['summary']['only_in_model1_count'] > 0 or
            results['summary']['only_in_model2_count'] > 0):
        return False, results


    all_identical = (results['summary']['identical_count'] ==
                     results['summary']['total_common'])

    return all_identical, results
