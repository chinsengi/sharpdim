from src.utils import load, load_data, load_net, calculateNeuronwiseHessians_fc_layer, load_fmnist, save_npy
from torch.nn import MSELoss
import json

def get_rel_flat(start_id, end_id, data="fashionmnist"):
    run_ids = [i for i in range(start_id, end_id)]
    n_sample = 100
    train_loader, _ = load_data(
            data, n_sample, batch_size=n_sample
        )
    for idx in run_ids:
        try:
            with open(f"./run/{data}/{idx}/config.json", "rb") as f:
                config = json.load(f)
            rel_flat_list = []
            model = load_net(config['network'], data, 10, config['nonlinearity']).to("cuda")
            load(f"./res/models/{data}{idx}.pkl", model)
        except Exception as e:
            print(e)
            continue
        print(idx)
        trace_nm, _ = calculateNeuronwiseHessians_fc_layer(model, train_loader, n_sample, MSELoss())
        rel_flat_list.append(trace_nm)
        breakpoint()
        save_npy(rel_flat_list, f"./res/{data}/{idx}", f"rel_flatness_list{idx}.npy")
        
        
if __name__ == "__main__":
    # data = input("Enter the dataset: ")
    # start_id = int(input("Enter the start id: "))
    # end_id = int(input("Enter the end id: "))
    data = 'cifar10'
    start_id = 100
    end_id = 200
    get_rel_flat(start_id, end_id, data)
    