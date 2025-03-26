start.dfs.sh
start-yarn.sh
mapred –daemon start history server
hadoop fs -mkdir -p /user/hadoop
jps
hadoop fs -mkdir -p /user/data/movielens/raw 
hadoop fs -mkdir /user/data/movielens/raw/users 
hadoop fs -mkdir /user/data/movielens/raw/movies 
hadoop fs -mkdir /user/data/movielens/raw/ratings
cd Exercises/ml-1m
hadoop fs -put users.dat /user/data/movielens/raw/users 
hadoop fs -put movies.dat /user/data/movielens/raw/movies 
hadoop fs -put ratings.dat /user/data/movielens/raw/ratings
hadoop fs -ls -R /user/data/movielens
./do_mapreduce.sh movies
./do_mapreduce.sh users
./do_mapreduce.sh ratings 
hive -f create_users_table.hql 
hive -f create_movies_table.hql 
hive -f create_ratings_table.hql
./do_mapreduce_mahout.sh
hadoop fs -put mahout_users.dat /user/data/movielens/mahout
./do_mahout.sh
hadoop fs -cat /user/data/movielens/mahout/output/part-r-00000