library(readxl)
library(psych)
library(GGally)
library(MVN)
library(psy)
library(dplyr)
library(magrittr)

# Membaca data dari file Excel
dat2 <- read_xlsx("C:/Riset/datapondasi.xlsx", sheet = 1)

dat2 <- dat2 %>%
  slice(-c(1, 2))

# Menghapus kolom teks yang tidak diperlukan
dat2 <- dat2 %>% select(5, 6, 7, 8, 9, 10)

# Mengonversi tanda koma desimal ke titik desimal dan mengonversi ke numerik
dat2 <- dat2 %>% 
  mutate(across(everything(), ~as.numeric(gsub(",", ".", .))))

# Korelasi data
R <- cor(dat2, use = "pairwise.complete.obs")
eigen <- eigen(R)
print(eigen)

# Memperoleh p-value dari matriks korelasi
corr.p(r = R, n = 5461)

# Visualisasi korelasi data
ggpairs(data = dat2)

# Kecukupan ukuran sampling, MSA
print(KMO(R))

# Melakukan Uji Bartlett pada matriks korelasi
print(cortest.bartlett(R, n = 5461))

# Mengambil sampel acak dari data (misalnya 5000 baris)
set.seed(123)
sampled_dat2 <- dat2 %>% sample_n(5000)

# Melakukan uji normalitas multivariat pada sampel
mvn_result <- mvn(data = sampled_dat2, mvnTest = "dh", univariateTest = "SW", multivariatePlot = "qq")
print(mvn_result)

# Pemeriksaan outliers menggunakan mvn() dengan parameter outlierMethod
outliers <- mvn(data = sampled_dat2, mvnTest = "hz", univariateTest = "SW", multivariateOutlierMethod = "adj", multivariatePlot = "qq")
print(outliers$multivariateOutliers)

# Menambahkan nilai konstan untuk menghindari log(0) dan log(nilai negatif)
sampled_dat2_adjusted <- sampled_dat2 %>%
  mutate(across(everything(), ~ . + abs(min(., na.rm = TRUE)) + 1))

# Transformasi data (log transformasi) setelah penyesuaian
transformed_dat2 <- sampled_dat2_adjusted %>%
  mutate(across(everything(), ~log1p(.)))

# Uji normalitas multivariat pada data yang telah ditransformasi
mvn_transformed_result <- mvn(data = transformed_dat2, mvnTest = "dh", univariateTest = "SW", multivariatePlot = "qq")
print(mvn_transformed_result)

# Korelasi pada data yang telah ditransformasi
R2 <- cor(transformed_dat2, use = "pairwise.complete.obs")
eigen2 <- eigen(R2)
print(eigen2)

# Memperoleh p-value dari matriks korelasi
corr.p(r = R2, n = 5461)

# Melakukan Uji Bartlett pada matriks korelasi
print(cortest.bartlett(R2, n = 5461))

# Melakukan analisis faktor
scree.plot(R2)
fa <- fa(r = R2, nfactors = 5, rotate = "oblimin", fm = "ml")
print(fa)
print(fa$communality)

# Jumlah faktor
plot(fa$values, type = "b", xlim = c(1, 10), main = "Scree plot",
     xlab = "Number of factors", ylab = "Eigen values")
abline(h = 1, lty = 2, col = "red")
