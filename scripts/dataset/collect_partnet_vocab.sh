Vocab="vase"
echo start to Collect Category:$Vocab Vocab For PartNet Dataset
/Users/melkor/miniforge3/envs/Melkor/bin/python datasets/structure_net/generate_structure_qa.py\
 --mode="geo" --category=$Vocab