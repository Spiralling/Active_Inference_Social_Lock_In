# Big sweep report

2455 successful runs.

## Health

- 248/2455 runs flagged diverged (max|Pi| > 1e3 or non-finite).
- diverged fraction by forgetting omega:
    omega=0.7: 0.00 (0/339)
    omega=0.8: 0.00 (0/340)
    omega=0.85: 0.00 (0/315)
    omega=0.9: 0.00 (0/811)
    omega=0.95: 0.11 (39/352)
    omega=1.0: 0.70 (209/298)

## Axis importance (Tier-2 random sample, diverged runs excluded)

- **final_struct_dist** most-moved-by: attention_beta(0.77), n_communities(0.71), inter(0.50), fuse_mode(0.14), omega(0.12)
- **lockin_frac** most-moved-by: omega(0.28), conviction_tilt(0.16), sigma_o(0.16), blend_ratio(0.13), world_mode(0.12)
- **final_comm_diversity** most-moved-by: conviction_tilt(0.37), omega(0.32), n_communities(0.30), sigma_o(0.28), blend_ratio(0.17)

## Flags (diverged runs excluded)

- max structural divergence 1.40 at N=60 inter=0.0 omega=0.95 n_comm=3 beta=100 world=changing_epochs.
- max lock-in 1.00 (changing-epochs) at omega=0.7 tilt=1.0 inter=0.1.

## Phase maps
- figures/phase_A_omega_inter_final_struct_dist.png
- figures/phase_A_omega_inter_final_comm_diversity.png
- figures/phase_A_omega_inter_lockin_frac.png
- figures/phase_A_omega_inter_max_pi.png
- figures/phase_B_inter_ncomm_final_struct_dist.png
- figures/phase_B_inter_ncomm_final_comm_diversity.png
- figures/phase_B_inter_ncomm_lockin_frac.png
- figures/phase_B_inter_ncomm_max_pi.png
- figures/phase_C_omega_tilt_final_struct_dist.png
- figures/phase_C_omega_tilt_final_comm_diversity.png
- figures/phase_C_omega_tilt_lockin_frac.png
- figures/phase_C_omega_tilt_max_pi.png
- figures/phase_D_N_inter_final_struct_dist.png
- figures/phase_D_N_inter_final_comm_diversity.png
- figures/phase_D_N_inter_lockin_frac.png
- figures/phase_D_N_inter_max_pi.png
- figures/phase_E_blend_beta_final_struct_dist.png
- figures/phase_E_blend_beta_final_comm_diversity.png
- figures/phase_E_blend_beta_lockin_frac.png
- figures/phase_E_blend_beta_max_pi.png
- figures/phase_F_sigma_omega_final_struct_dist.png
- figures/phase_F_sigma_omega_final_comm_diversity.png
- figures/phase_F_sigma_omega_lockin_frac.png
- figures/phase_F_sigma_omega_max_pi.png
- figures/phase_G_fuse_inter_final_struct_dist.png
- figures/phase_G_fuse_inter_final_comm_diversity.png
- figures/phase_G_fuse_inter_lockin_frac.png
- figures/phase_G_fuse_inter_max_pi.png
