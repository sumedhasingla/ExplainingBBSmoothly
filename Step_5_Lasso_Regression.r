#!/usr/bin/env Rscript

library("optparse")
library(glmnet)
library(readr)
library(caret)
seed = sample(1:100, 1, replace=T)
set.seed(seed)

option_list = list(
  make_option(c("-c", "--concept_dir"), type="character", default='', 
              help="directory where concept files are saved", metavar="character"),
  make_option(c("-d", "--file_name"), type="character", default='feature_dense_4_interstitial_marking_max_', 
              help="file name suffix", metavar="character"),
  make_option(c("-m", "--measure"), type="character", default='auc', 
              help="type [default= %default]", metavar="character")
)

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
if (is.null(opt$concept_dir)){
  print_help(opt_parser)
  stop("concept_dir must be supplied", call.=FALSE)
}
if (is.null(opt$file_name)){
  print_help(opt_parser)
  stop("file_name must be supplied", call.=FALSE)
}
if (is.null(opt$measure)){
  print_help(opt_parser)
  stop("measure must be supplied", call.=FALSE)
}

concept_dir = opt$concept_dir
file_name = opt$file_name
measure = opt$measure
print(paste("file_name: ", file_name, "measure: ", measure))


output_file_name = paste(concept_dir, '/',file_name,'_output_R.txt',sep='')

if (file.exists(output_file_name)){
stop("Already completed", call.=FALSE)
}

data_present = read_csv(paste(concept_dir, '/', file_name,'_positive.csv',sep=''), col_names=FALSE)
data_absent = read_csv(paste(concept_dir, '/',  file_name,'_negative.csv',sep=''), col_names=FALSE)
data_present$label = rep(1, dim(data_present)[1])
data_absent$label = rep(0, dim(data_absent)[1])
dim(data_present)
dim(data_absent)
data = rbind(data_present, data_absent)
dim(data)

# Now Selecting 75% of data as sample from total 'n' rows of the data  
sample <- sample.int(n = nrow(data), size = floor(.75*nrow(data)), replace = F)
train <- data[sample, ]
test  <- data[-sample, ]
dim(train)
dim(test)
column_names = names(train)[1:ncol(train)-1]

best_alpha = 1
sink(output_file_name)
print(seed)

#Repeat lasso regression 10 times
features = list()
features_count = rep(0, 10) 
for (i in 1:10){
    print(i)

    fit =  cv.glmnet(as.matrix(train[column_names]), train$label, type.measure="auc", alpha=1,family="binomial", nfolds=5)
    imp_features_cv_1se_current = rownames(coef(fit, s = 'lambda.1se'))[coef(fit, s = 'lambda.1se')[,1]!= 0] 
    features$i = imp_features_cv_1se_current
    features_count[i] = length(imp_features_cv_1se_current)
    
    print(fit)

    pre = predict(fit, newx = as.matrix(test[column_names]), s = c(fit$lambda.min, fit$lambda.1se))
    print(assess.glmnet(pre, newy = test$label, family = "binomial"))
    print(confusion.glmnet(fit, newx = as.matrix(test[column_names]), test$label, family = "binomial"))
    
    pre = predict(fit, newx = as.matrix(data[column_names]), s = c(fit$lambda.min, fit$lambda.1se))
    print(assess.glmnet(pre, newy = data$label, family = "binomial"))
    print(confusion.glmnet(fit, newx = as.matrix(data[column_names]), data$label, family = "binomial"))

    print("Important Features Coeff - Min Lambda")
    imp_features_cv_min =coef(fit, s = 'lambda.min')[coef(fit, s = 'lambda.min')[,1]!= 0] ### returns nonzero coefs
    print(imp_features_cv_min)

    print("Important Features Coeff - 1se Lambda")
    imp_features_cv_1se =coef(fit, s = 'lambda.1se')[coef(fit, s = 'lambda.1se')[,1]!= 0] 
    print(imp_features_cv_1se)

    print("Important Features - Min Lambda")
    imp_features_cv_min = rownames(coef(fit, s = 'lambda.min'))[coef(fit, s = 'lambda.min')[,1]!= 0] ### returns nonzero coefs
    print(imp_features_cv_min)

    print("Important Features - 1se Lambda")
    imp_features_cv_1se = rownames(coef(fit, s = 'lambda.1se'))[coef(fit, s = 'lambda.1se')[,1]!= 0] 
    print(imp_features_cv_1se)
}


#Run a stable feature selection as well
library('stabs')
stab.glmnet <- stabsel(x = as.matrix(train[column_names]), y = train$label, fitfun = glmnet.lasso, args.fitfun = list(type.measure="auc", alpha=1,family="binomial"), cutoff = 0.75,PFER = 1)

print(stab.glmnet)


#After fitting lasso for 10 times:
print("feature counts in 10 runs of lasso: ")
print(features_count)
mean_num_features = mean(features_count)
std_num_features = sd(features_count)
flag = 0
for (i in 1:10){
    if (features_count[i] >  mean_num_features - (std_num_features/2)){
        if (features_count[i] <  mean_num_features + (std_num_features/2)){
            print(i)
            if (flag ==0){
            common_feature = features$i
            }
            else{
            common_feature = union(common_feature, features$i)
            }
        }
    }
}

print("Final Features:")
print(common_feature)
sink()

