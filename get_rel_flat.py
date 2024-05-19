from regex import W
from src.utils import load, load_data, load_net, calculateNeuronwiseHessians_fc_layer, load_fmnist, save_npy
from torch.nn import MSELoss
import json
from src.utils import get_nmls

def get_rel_flat(start_id, end_id, data="fashionmnist"):
    run_ids = [i for i in range(start_id, end_id)]
    n_sample = 100
    train_loader, _ = load_data(
            data, n_sample, batch_size=1
        )
    for idx in run_ids:
        try:
            with open(f"./run/{data}/{idx}/config.json", "rb") as f:
                config = json.load(f)
            model = load_net(config['network'], data, 10, config['nonlinearity']).to("cuda")
            load(f"./res/models/{data}{idx}.pkl", model)
            rel_flat_list = []
            harm_list = []
            W_list = []
            W0_list = []
            # trace_nm, _ = calculateNeuronwiseHessians_fc_layer(model, train_loader, n_sample, MSELoss())
            # rel_flat_list.append(trace_nm)
            nmls, harmonic,W0_norm,W_norm = get_nmls(model, train_loader, n_sample)
            harm_list.append(harmonic.item() )
            W_list.append(W_norm.item())
            W0_list.append(W0_norm.item())
            # save_npy(rel_flat_list, f"./res/{data}/{idx}", f"rel_flatness_list{idx}.npy")
            save_npy(harm_list, f"./res/{data}/{idx}", f"harm_list{idx}.npy")
            save_npy(W_list, f"./res/{data}/{idx}", f"W_list{idx}.npy")
            save_npy(W0_list, f"./res/{data}/{idx}", f"W0_list{idx}.npy")
            # breakpoint()
        except Exception as e:
            # save_npy(harm_list, f"./res/{data}/{idx}", f"harm_list{idx}.npy")
            print(e)
            continue
        print(idx)
        
        
if __name__ == "__main__":
    # data = input("Enter the dataset: ")
    # start_id = int(input("Enter the start id: "))
    # end_id = int(input("Enter the end id: "))
    data = 'cifar10'
    # data = 'fashionmnist'
    start_id = 400
    end_id = 500
    get_rel_flat(start_id, end_id, data)
    