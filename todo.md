
[] filter / pad / etc. cases of short < k user sets of scores.
[] setup hparam sweeps
  [x] update config to dataclass
  [] take class as arg instead of import
  [] add hparam sweep script
[] optimize for training kandinsky prior (may need grad checkpointing, offloading, etc)
  [] don't keep the kandinsky decoder pipeline on device; offload or del each val.
[] diffusion vs eclipse-style MSE ablation.
  [] almost certain diffusion will be preferable but checking is fast + worthwhile.
  [] sweep diffusion noise schedule? 
  [] try flow matching?
[x] overhaul app.py demo
  - multi-user is definitely broken
  [] add back multi-user functionality?
[] panel for individual users from data
  [] and then from their history -> generation


Done:
[x] unify the qual_val and app.py code to use prior pipeline
[x] add diffusion training
  [x] test changes
  [x] update loss calc
  [x] update prior pipe
  [x] add back timestep conditioning
[x] pretrain PoC diffusion model
[x] light standalone inference.py

