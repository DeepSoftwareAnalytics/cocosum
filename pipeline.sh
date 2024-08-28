echo "Step 0: Download original csn (CodeSearchNet) data"
mkdir -p data/csn
cd data/csn
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
unzip -o java.zip

echo "Step 1: Download git repository"
cd ../../datapreprocess/CodeSearchNet/
mkdir log
python 1_get_git_repository.py |tee ./log/1_get_git_repository.txt

echo "Step 2: Add fid to original csn data"
python 2_add_id.py |tee ./log/2_add_id.txt

echo "Step 3: Token preprocessing for code, summary, and sbt."
#sudo dpkg -i srcML-Ubuntu14.04.deb
#if [ $? -ne 0 ]; then
#  echo "Please ignore the error information about srcml installation"
#  sudo dpkg -i srcML-Ubuntu18.04.deb
#fi
python 3_token_preprocessing.py |tee ./log/3_token_preprocessing.txt

echo "Step 4: Extract uml"

echo "Step 4.1: Split large dataset to a set of small datasets"
python 4_1_split_dataset.py |tee ./log/4_1_split_dataset.txt

echo "Step 4.2: Generate dot using UmlGraph"
python 4_2_get_dot_with_umlgraph.py |tee ./log/4_2_get_dot_with_umlgraph.txt

echo "Step 4.3:  Parse dot and get uml "
python 4_3_get_uml.py |tee   ./log/4_3_get_uml.txt

echo "Step 4.4: Merge all small datasets "
python 4_4_merge_m2uid.py |tee   ./log/4_4_merge_m2uid.txt

echo "Step 4.5: Filter umls"
python 4_5_filter_class_not_in_uml.py |tee   ./log/4_5_filter_class_not_in_uml.txt

echo "Step 5: Uml embedding"
python 5_uml_embedding.py |tee ./log/5_uml_embedding.txt

echo "Step 6: Get correct fid"
python 6_get_correct_id.py |tee ./log/6_get_correct_id.txt

echo "Step 7: word count and build vocabulary"
python 7_build_vocabulary.py |tee ./log/7_build_vocabulary.txt

echo "Step 8: build dataset"
python 8_build_dataset.py |tee ./log/8_build_dataset.txt

cd ../../

echo "Step 9: train"
cd ./
python runGruUml.py -modeltype=uml  -dty=ASTAttGRU_AttTwoChannelTrans |tee train.txt

echo "Step 10: evaluation"
python evaluate.py -modeltype=uml -model_root_path="./model/" -model_file='model.pth' -prediction_path='./predict'