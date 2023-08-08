echo train the phase 1 of CSQNet on StructureNet
/Users/melkor/miniforge3/envs/Melkor/bin/python\
 train.py --name="VNL"\
 --dataset="StructureNet" --perception="csqnet" --training_mode="3d_perception" \
  --phase="1" --batch_size=1 --concept_type="cone" \
 --freeze_perception="True" --lr="0.0002"\
  --checkpoint_dir="checkpoints/VNL_3d_perception_structure_csqnet_phase0.pth"