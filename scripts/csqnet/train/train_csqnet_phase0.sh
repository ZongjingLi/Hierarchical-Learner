echo train the phase 0 of CSQNet on StructureNet
/Users/melkor/miniforge3/envs/Melkor/bin/python\
 train.py --dataset="StructureNet" --perception="csqnet" --training_mode="3d_perception" \
  --phase="0" --batch_size=2 --concept_type="box" \
  