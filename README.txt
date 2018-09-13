# Problem Pointing System Based on Opinion Mining from Social Media

## Dekripsi
Pada penelitian ini, dibangun problem pointing system dengan memanfaatkan opini masyarakat pada media sosial. Sistem yang dibangun digunakan untuk mengolah dan menyajikan opini masyarakat.

## File pada projek
1. File CrawlerTwitter.py berfungsi untuk melakukan crawling data dari Twitter
3. File ProblemPointingSystem.py berfungsi sebagai fungsi utama yang akan dijalankan pada sistem

## Bagaimana cara memulai
Projek ini menggunakan python 3.5 sebagai sistem build dan windows sebagai sistem operasi

Langkah pertama yang harus didownload ialah python versi 3.5
Anda dapat mendownload pada link berikut : https://www.python.org/download/releases/3.5/

Projek ini dapat dibuka dalam beberapa tools yakni : Sublime, Anaconda, Spider, dll. Peneliti menggunakan sublime dalam menjalankan projek ini.

Setelah itu anda perlu menginstall modul-modul yang diperlukan untuk menjalan sistem. yakni :
1.  cv2
2.  sys
3.  nltk
3.  string
4.  re
5.  pickle
6.  csv
7.  codecs
8.  unicodedata
9.  os
10. pytorch
11. pandas
12. numpy
13. matplotlib
14. seaborn
15. tweepy

Modul diatas dapat didownload dengan sintaks 'pip install < nama_module >'

## Run Program
1. Untuk crawling data, run program 'CrawlerTwitter.py' pada terminal command prompt
2. Untuk run program dapat menggunakan 'python ProblemPointingSystem.py' pada terminal command prompt.
3. Setelah GUI telah muncul tekan tombol 'Read CSV File...' kemudian pilih data yang akan diproses
4. Kemudian tekan tombol 'Start', maka akan dilakukan semua proses dan ditampilkan semua hasil