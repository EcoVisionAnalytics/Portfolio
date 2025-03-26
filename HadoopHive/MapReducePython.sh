curl https://www.gutenberg.org/files/1228/1228-0.txt
sed '1,318d' darwin.txt > darwin_3.txt 
sed '14637,16183d' darwin_3.txt > darwin_1.txt 
counts3 < darwin1.txt  
mapper3.py < darwin1.txt
mapper3.py < darwin1.txt|sort
mapper3.py <darwin1.txt|sort|reducer3.py 
tr -d '[:punct:]' < darwin1.txt > darwin_clean1.txt 
tr [:upper:][:lower:] < darwin_clean1.txt > darwinclean.txt
mapper3.py <darwin_clean1.txt|sort|reducer3.py
