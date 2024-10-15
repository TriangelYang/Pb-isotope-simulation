import numpy as np
import torch
from tqdm import tqdm

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
lambda1 = torch.tensor(1.5513e-10, dtype=torch.float32, device=device)
lambda2 = torch.tensor(9.8485e-10, dtype=torch.float32, device=device)

# data
t2 = torch.tensor([3687800000, 3130100000, 2030000000], dtype=torch.float32, device=device)
t2_err = torch.tensor([4500000, 9800000, 4000000], dtype=torch.float32, device=device)
Pb6_Pb4 = torch.tensor([104.47774326, 247.50170188, 439.7248796 ], dtype=torch.float32, device=device)
Pb6_Pb4_err = torch.tensor([2.05848569, 3.45132842, 14.73440979], dtype=torch.float32, device=device)
Pb7_Pb6 = torch.tensor([1.399, 1.1516, 0.86], dtype=torch.float32, device=device)
Pb7_Pb6_err = torch.tensor([0.020, 0.0048, 0.019], dtype=torch.float32, device=device)

num_vars = 3
num_samples = 2000
T_range = torch.arange(4300000000, 4530000000, 3000000, dtype=torch.float32, device=device)
t1_range = torch.arange(4340000000, 4500000000, 2000000, dtype=torch.float32, device=device)
num_T = len(T_range)
num_t1 = len(t1_range)
all_sim_times = 100000000

# list for saving results
mu_all = []
t1_all = []
T_all = []
LMO_46_all = []
LMO_76_all = []
LMO_64_all = []
LMO_74_all = []

mu_range_all = []
LMO_64_range_all = []
LMO_74_range_all = []

# start simulating
for _ in tqdm(range(all_sim_times // num_samples)):
    target_shape = (num_T, num_t1, num_samples, num_vars)
    T_samples = T_range[:, None, None, None].expand(target_shape)
    t1_samples = t1_range[None, :, None, None].expand(target_shape)
    
    # sample from normal distribution
    t2_samples = (torch.randn((num_T, num_t1, num_samples, num_vars), device=device) * (t2_err / 2)) + t2
    Pb6_Pb4_samples = (torch.randn((num_T, num_t1, num_samples, num_vars), device=device) * (Pb6_Pb4_err / 2)) + Pb6_Pb4
    Pb7_Pb6_samples = (torch.randn((num_T, num_t1, num_samples, num_vars), device=device) * (Pb7_Pb6_err / 2)) + Pb7_Pb6

    # # sample from uniform distribution
    # t2_samples = (torch.rand((num_T, num_t1, num_samples, num_vars), device=device) * t2_err * 2) + (t2 - t2_err)
    # Pb6_Pb4_samples = (torch.rand((num_T, num_t1, num_samples, num_vars), device=device) * Pb6_Pb4_err * 2) + (Pb6_Pb4 - Pb6_Pb4_err)
    # Pb7_Pb6_samples = (torch.rand((num_T, num_t1, num_samples, num_vars), device=device) * Pb7_Pb6_err * 2) + (Pb7_Pb6 - Pb7_Pb6_err)

    Pb7_Pb4_samples = Pb7_Pb6_samples * Pb6_Pb4_samples

    exp_lambda1_T4567 = torch.exp(lambda1 * torch.tensor(4567000000.0, dtype=torch.float32, device=device))
    exp_lambda2_T4567 = torch.exp(lambda2 * torch.tensor(4567000000.0, dtype=torch.float32, device=device))
    
    mu = ((Pb6_Pb4_samples - (9.307 + 1.8 * (exp_lambda1_T4567 - torch.exp(lambda1 * T_samples)))) * (torch.exp(lambda2 * t1_samples) - torch.exp(lambda2 * t2_samples)) - \
        (Pb7_Pb4_samples - (10.294 + 1.8 / 137.818 * (exp_lambda2_T4567 - torch.exp(lambda2 * T_samples)))) * 137.818 * (torch.exp(lambda1 * t1_samples) - torch.exp(lambda1 * t2_samples))) / \
        ((torch.exp(lambda1 * T_samples) - torch.exp(lambda1 * t1_samples)) * (torch.exp(lambda2 * t1_samples) - torch.exp(lambda2 * t2_samples)) - \
        (torch.exp(lambda2 * T_samples) - torch.exp(lambda2 * t1_samples)) * (torch.exp(lambda1 * t1_samples) - torch.exp(lambda1 * t2_samples)))
    
    Pb6_Pb4_LMO = 9.307 + 1.8 * (exp_lambda1_T4567 - torch.exp(lambda1 * T_samples)) + mu * (torch.exp(lambda1 * T_samples) - torch.exp(lambda1 * t1_samples))
    Pb7_Pb4_LMO = 10.294 + 1.8 / 137.818 * (exp_lambda2_T4567 - torch.exp(lambda2 * T_samples)) + mu / 137.818 * (torch.exp(lambda2 * T_samples) - torch.exp(lambda2 * t1_samples))

    # define the range limitation of the samples as a rough screening of calculation results
    # the range of three samples should not be greather than 380 (adjustable)
    max_ptp = [380]

    # All μ1 greater than 0
    filter_condition_1 = torch.any(mu > 0, dim=-1, keepdim=True)
    # filter_condition_1 &= torch.any(mu < 800, dim=-1, keepdim=True)
    # # T greater than t1
    # filter_condition_2 = T_samples != t1_samples
    filter_condition_2 = T_samples != 0
    filter_condition = filter_condition_1 & filter_condition_2

    mu_filtered = mu[filter_condition].view(-1, 3)
    t1_filtered = t1_samples[filter_condition].view(-1, 3)
    T_filtered = T_samples[filter_condition].view(-1, 3)
    Pb6_Pb4_LMO_filtered = Pb6_Pb4_LMO[filter_condition].view(-1, 3)
    Pb7_Pb4_LMO_filtered = Pb7_Pb4_LMO[filter_condition].view(-1, 3)

    # calculate the rage of each triplet manually
    mu_range = mu_filtered.max(dim=1)[0] - mu_filtered.min(dim=1)[0]
    # mu_range = mu_filtered.std(dim=1)
    # mu_range = torch.abs(mu_filtered[:, 2] - mu_filtered[:, 1])
    Pb6_Pb4_LMO_range_01 = torch.abs(Pb6_Pb4_LMO_filtered[:, 0] - Pb6_Pb4_LMO_filtered[:, 1])
    Pb6_Pb4_LMO_range_02 = torch.abs(Pb6_Pb4_LMO_filtered[:, 0] - Pb6_Pb4_LMO_filtered[:, 2])
    Pb6_Pb4_LMO_range_12 = torch.abs(Pb6_Pb4_LMO_filtered[:, 1] - Pb6_Pb4_LMO_filtered[:, 2])

    for p in max_ptp:
        # print(p)
        _ptp_filter = (mu_range < p) & (Pb6_Pb4_LMO_range_02 < 0.000005)
        # _ptp_filter = (mu_range < p)
        ptp_filter = _ptp_filter.unsqueeze(-1).expand_as(mu_filtered)

        mu_filtered_twice = mu_filtered[ptp_filter].view(-1, 3).mean(dim=1)
        t1_filtered_twice = t1_filtered[ptp_filter].view(-1, 3).mean(dim=1)
        T_filtered_twice = T_filtered[ptp_filter].view(-1, 3).mean(dim=1)
        Pb6_Pb4_LMO_filtered_twice = Pb6_Pb4_LMO_filtered[ptp_filter].view(-1, 3)[:, [0]]
        Pb7_Pb4_LMO_filtered_twice = Pb7_Pb4_LMO_filtered[ptp_filter].view(-1, 3)[:, [0]]
        Pb4_Pb6_LMO_filtered_twice = 1 / Pb6_Pb4_LMO_filtered_twice
        Pb7_Pb6_LMO_filtered_twice = Pb7_Pb4_LMO_filtered_twice * Pb4_Pb6_LMO_filtered_twice

        mu_all.append(mu_filtered_twice.cpu().numpy())
        T_all.append(T_filtered_twice.cpu().numpy())
        t1_all.append(t1_filtered_twice.cpu().numpy())
        LMO_46_all.append(Pb4_Pb6_LMO_filtered_twice.cpu().numpy())
        LMO_76_all.append(Pb7_Pb6_LMO_filtered_twice.cpu().numpy())
        mu_range_all.append(mu_range[_ptp_filter].cpu().numpy())
        LMO_64_range_all.append(Pb6_Pb4_LMO_filtered_twice.cpu().numpy())
        LMO_74_range_all.append(Pb7_Pb4_LMO_filtered_twice.cpu().numpy())
        LMO_64_all.append(Pb6_Pb4_LMO_filtered_twice.mean(dim=1).cpu().numpy())
        LMO_74_all.append(Pb7_Pb4_LMO_filtered_twice.mean(dim=1).cpu().numpy())

# concatenate all results into a one-dimensional array
mu_all_array = np.concatenate(mu_all)
T_all_array = np.concatenate(T_all)
t1_all_array = np.concatenate(t1_all)
LMO_46_all_array = np.concatenate(LMO_46_all)
LMO_76_all_array = np.concatenate(LMO_76_all)
LMO_64_all_array = np.concatenate(LMO_64_all)
LMO_74_all_array = np.concatenate(LMO_74_all)

bin_size = 10
bin_edges = np.arange(mu_all_array.min(), mu_all_array.max() + bin_size, bin_size)
hist, bin_edges = np.histogram(mu_all_array, bins=bin_edges)
low_count_bins = np.where(hist < 3000)[0]
indices_to_remove = []
for bin_index in low_count_bins:
    bin_start = bin_edges[bin_index]
    bin_end = bin_edges[bin_index + 1]
    indices_to_remove.extend(np.where((mu_all_array >= bin_start) & (mu_all_array < bin_end))[0])

mu_all_array_filtered = np.delete(mu_all_array, indices_to_remove)
T_all_array_filtered = np.delete(T_all_array, indices_to_remove)
t1_all_array_filtered = np.delete(t1_all_array, indices_to_remove)
LMO_46_all_array_filtered = np.delete(LMO_46_all_array, indices_to_remove)
LMO_76_all_array_filtered = np.delete(LMO_76_all_array, indices_to_remove)
LMO_64_all_array_filtered = np.delete(LMO_64_all_array, indices_to_remove)
LMO_74_all_array_filtered = np.delete(LMO_74_all_array, indices_to_remove)

if len(mu_all_array_filtered) == 0:
    print("no solution")
    print("----------------------------------------------------------------------------")

print("mu1 = {} ± {}".format(mu_all_array_filtered.mean(), mu_all_array_filtered.std() * 2))
print("t1 = {} ± {}".format(t1_all_array_filtered.mean(), t1_all_array_filtered.std() * 2))
print("T = {} ± {}".format(T_all_array_filtered.mean(), T_all_array_filtered.std() * 2))
print("Pb4/Pb6_LMO = {} ± {}".format(LMO_46_all_array_filtered.mean(), LMO_46_all_array_filtered.std() * 2))
print("Pb7/Pb6_LMO = {} ± {}".format(LMO_76_all_array_filtered.mean(), LMO_76_all_array_filtered.std() * 2))
print("Pb6/Pb4_LMO = {} ± {}".format(LMO_64_all_array_filtered.mean(), LMO_64_all_array_filtered.std() * 2))
print("Pb7/Pb4_LMO = {} ± {}".format(LMO_74_all_array_filtered.mean(), LMO_74_all_array_filtered.std() * 2))
print("")
