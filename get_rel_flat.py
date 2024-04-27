from src.utils import load, load_net, calculateNeuronwiseHessians_fc_layer, load_fmnist, save_npy
from torch.nn import MSELoss

def get_rel_flat():
    run_ids = [i for i in range(61, 81)]
    for idx in run_ids:
        rel_flat_list = []
        try:
            model = load_net("fnn", "fashionmnist", 10, "relu").to("cuda")
            load(f"./res/models/fashionmnist{idx}.pkl", model)
        except:
            continue
        print(idx)
        n_sample = 100
        train_loader, _ = load_fmnist(
            n_sample, batch_size=n_sample
        )
        trace_nm, _ = calculateNeuronwiseHessians_fc_layer(model, train_loader, n_sample, MSELoss())
        rel_flat_list.append(trace_nm)
        save_npy(rel_flat_list, f"./res/fashionmnist/{idx}", f"rel_flat_list{idx}.npy")
        
        
if __name__ == "__main__":
    get_rel_flat()
    