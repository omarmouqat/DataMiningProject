

hdfs dfs -rm -r /output
hdfs dfs -rm -r /dataset
hdfs dfs -mkdir /dataset
hdfs dfs -put /home/import/proce_train.csv /dataset/
hadoop jar /home/import/target/hadoopproject-1.0-SNAPSHOT.jar Application