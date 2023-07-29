echo train the phase 0 of CSQNet on StructureNet
/Users/melkor/miniforge3/envs/Melkor/bin/python\
 train.py --name="VNL"\
 --dataset="StructureNet" --perception="csqnet" --training_mode="3d_perception" \
  --phase="1" --batch_size=2 --concept_type="box" \
 --checkpoint_dir="checkpoints/VNL_3d_perception_toy_csqnet_phase0.pth"