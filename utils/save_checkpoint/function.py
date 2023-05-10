import torch


# 保存检查点：参数为要保存的项，可根据需要增减，最后checkpoint_path参数是保存路径
def save_checkpoint(model_, checkpoint_path):
    save_dict = {
        "model_state_dict": model_.state_dict()  # 模型参数
    }
    torch.save(save_dict, checkpoint_path)
