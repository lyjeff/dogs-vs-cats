import torch
from models.model import MyCNN
from models.model import ExampleCNN
from datasets.dataloader import make_test_dataloader

import os
from tqdm import tqdm

def test(model_name, device, base_path, save_path):
    test_data_path = os.path.join(base_path, "data", "test")
    weight_path = os.path.join(save_path, "weight.pth")

    # load model and use weights we saved before
    if model_name == "ExampleCNN":
        model = ExampleCNN()
    else:
        model = MyCNN()

    model.load_state_dict(torch.load(weight_path))
    model = model.to(device)

    # make dataloader for test data
    test_loader = make_test_dataloader(test_data_path, 2)

    predict_correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)

            output = model(data)
            predict_correct += (output.data.max(1)[1] == target.data).sum()

        accuracy = 100. * predict_correct / len(test_loader.dataset)
    print(f'Test accuracy: {accuracy:.4f}%')

    return accuracy.item()

if __name__ == '__main__':
    cuda_device = 0
    batch_size =32
    epochs = 40
    learning_rate = 0.01
    model_name = "ExampleCNN"

    base_path = os.path.dirname(os.path.abspath(__file__))
    device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')

    state_name = f"{batch_size}_{epochs}_{learning_rate}"
    save_name = "train_result"

    save_path = os.path.join(base_path, save_name, state_name)

    if not os.path.exists(os.path.join(base_path, save_name)):
        os.mkdir(os.path.join(base_path, save_name))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    test_accuracy = test(
        model_name=model_name,
        device=device,
        base_path=base_path,
        save_path=save_path
    )