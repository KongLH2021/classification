#import data set
hotels<- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/hotels.csv")
summary(hotels)

library("skimr")
skim(hotels)

require('DataExplorer')
DataExplorer::plot_bar(hotels, ncol = 3)

library("tidyverse")
library("ggplot2")

ggplot(hotels,
       aes(x = is_canceled, y = stays_in_weekend_nights+stays_in_week_nights)) +
  geom_point()

hotels <- hotels %>%
  mutate(kids = case_when(
    children + babies > 0 ~ 'kids',
    TRUE ~ 'none'
  ))
hotels <- hotels %>%
  mutate(num.kids = case_when(
    children + babies > 0 ~ (children + babies),
    TRUE ~ 0
  ))
hotels <- hotels %>%
  select(-babies, -children)

view(hotels)

hotels <- hotels %>% 
  mutate(parking = case_when(
    required_car_parking_spaces > 0 ~ 'parking',
    TRUE ~ 'none'
  ))

library("GGally")

ggpairs(hotels %>% select(kids, adr, parking, total_of_special_requests),
        aes(color = kids))

hotels.bycountry <- hotels %>% 
  group_by(country) %>% 
  summarise(total = n(),
            cancellations = sum(is_canceled),
            pct.cancelled = cancellations/total*100)

ggplot(hotels.bycountry %>% arrange(desc(total)) %>% head(10),
       aes(x = country, y = pct.cancelled)) +
  geom_col()

library("rnaturalearth")
library("rnaturalearthdata")
library("rgeos")

world <- ne_countries(scale = "medium", returnclass = "sf")

world2 <- world %>%
  left_join(hotels.bycountry,
            by = c("iso_a3" = "country"))
ggplot(world2) +
  geom_sf(aes(fill = pct.cancelled))

hotels.par <- hotels %>%
  select(hotel, is_canceled, kids, meal, customer_type) %>%
  group_by(hotel, is_canceled, kids, meal, customer_type) %>%
  summarize(value = n())

library("ggforce")

ggplot(hotels.par %>% gather_set_data(x = c(1, 3:5)),
       aes(x = x, id = id, split = y, value = value)) +
  geom_parallel_sets(aes(fill = as.factor(is_canceled)),
                     axis.width = 0.1,
                     alpha = 0.66) + 
  geom_parallel_sets_axes(axis.width = 0.15, fill = "lightgrey") + 
  geom_parallel_sets_labels(angle = 0) +
  coord_flip()


install.packages('DataExplorer')
DataExplorer::plot_bar(hotels, ncol = 3)
DataExplorer::plot_histogram(hotels, ncol = 3)
DataExplorer::plot_boxplot(hotels, by = "is_canceled", ncol = 3)

hotels <- hotels %>% 
  select(-arrival_date_year, -meal, -country, -market_segment, -distribution_channel, -deposit_type, -agent, -company, -days_in_waiting_list, -customer_type, -reservation_status, -reservation_status_date, -assigned_room_type, -arrival_date_month, -hotel, -reserved_room_type, -kids, -parking)

summary(hotels)

library("data.table")
library("mlr3verse")

#change "is_canceled" from num 2 factor
hotels$is_canceled <- factor(hotels$is_canceled,levels = 0:1)

set.seed(1)
hotels_task <- TaskClassif$new(id = "hotel_canceled",
                               backend = hotels,
                               target = "is_canceled",
                               positive = "1")

#train/test/validate
train_test_set = sample(hotels_task$row_ids, 0.75 * hotels_task$nrow)
validate_set = setdiff(hotels_task$row_ids, train_test_set)
train_set = sample(hotels_task$row_ids, 0.7 * hotels_task$nrow)
test_set = setdiff(hotels_task$row_ids, train_set)

length(train_test_set)
length(validate_set)
length(train_set)
length(test_set)

#cv5
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(hotels_task)

#Logistic regression
#initial 
lrn_logreg = lrn("classif.log_reg", predict_type = "prob")
lrn_logreg$train(hotels_task, row_ids = train_set)
pred_logreg = lrn_logreg$predict(hotels_task, row_ids = test_set)
lrn_logreg$predict(hotels_task, row_ids = test_set)
#Performance Evaluation
resampling = rsmp("cv", folds=5)
res_logreg = resample(hotels_task, learner = lrn_logreg, resampling = resampling)
res_logreg$aggregate()#the lower the better

#install.packages('rpart')
require('rpart')

lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_baseline$train(hotels_task, row_ids = train_set)
pred_baseline = lrn_baseline$predict(hotels_task, row_ids = test_set)
lrn_baseline$predict(hotels_task, row_ids = test_set)
res_baseline <- resample(hotels_task, lrn_baseline, cv5, store_models = TRUE)
res_baseline$aggregate()


lrn_cart <- lrn("classif.rpart", predict_type = "prob", xval = 10 )
lrn_cart$train(hotels_task, row_ids = train_set)
pred_cart = lrn_cart$predict(hotels_task, row_ids = test_set)
lrn_cart$predict(hotels_task, row_ids = test_set)
res_cart <- resample(hotels_task, lrn_cart, cv5, store_models = TRUE)
rpart::plotcp(res_cart$learners[[5]]$model)
res_cart$aggregate()

# when cp = 0, the classif.ce gets the smallest value 0.1997655 
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.01, id = "cartcp")
lrn_cart_cp$train(hotels_task, row_ids = train_set)
pred_cart_cp = lrn_cart_cp$predict(hotels_task, row_ids = test_set)
lrn_cart_cp$predict(hotels_task, row_ids = test_set)
res_cart_cp <- resample(hotels_task, lrn_cart_cp, cv5, store_models = TRUE)
res_cart_cp$aggregate()


require('xgboost')
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
lrn_xgboost$train(hotels_task, row_ids = train_set)
pred_xgboost = lrn_xgboost$predict(hotels_task, row_ids = test_set)
lrn_xgboost$predict(hotels_task, row_ids = test_set)
res_xgboost <- resample(hotels_task, lrn_xgboost, cv5, store_models = TRUE)
res_xgboost$aggregate()

require('ranger')
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_ranger$train(hotels_task, row_ids = train_set)
pred_ranger = lrn_ranger$predict(hotels_task, row_ids = test_set)
lrn_ranger$predict(hotels_task, row_ids = test_set)
res_ranger <- resample(hotels_task, lrn_xgboost, cv5, store_models = TRUE)
res_ranger$aggregate()

'
#source from https://zhuanlan.zhihu.com/p/441604607
search_space = ps(
  mtry = p_int(lower = 1, upper = 16),
  num.trees = p_int(lower = 500, upper = 1000),
  min.node.size = p_int(lower = 2, upper = 10))

at = auto_tuner(
  learner = lrn_ranger,
  resampling = rsmp("cv", folds = 3),
  measure = msr("classif.auc"),
  search_space = search_space,
  method = "random_search",
  term_evals = 5)

at$train(hotels_task, row_ids = validate_set)

lrn_ranger$param_set$values = at$tuning_result$learner_param_vals[[1]]
lrn_ranger$train(hotels_task, row_ids = train_set)
lrn_ranger$predict(hotels_task, row_ids = test_set)
res_ranger <- resample(hotels_task, lrn_xgboost, cv5, store_models = TRUE)
res_ranger$aggregate()
'

res <- benchmark(data.table(
  task       = list(hotels_task),
  learner    = list(lrn_logreg,
                    lrn_baseline,
                    lrn_cart,
                    lrn_cart_cp,
                    lrn_xgboost,
                    lrn_ranger
  ),
  resampling = list(cv5)
), store_models = TRUE)
res$aggregate()
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.fpr"),
                   msr("classif.fnr"),
                   msr("classif.tnr"),
                   msr("classif.tpr"),
                   msr("classif.auc"),
                   msr("classif.logloss")
))


library("data.table")
library("mlr3verse")
library("tidyverse")

# base learners
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0, id = "cartcp")

# super learner
super_logreg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

pl_factor <- po("encode")

super_lrn <- gunion(list(
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart),
    po("learner_cv", lrn_cart_cp)
  )),
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg),
      po("nop")
    )),
  pl_factor %>>%
    po("learner_cv", lrn_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(super_logreg)


# Finally fit the base learners and super learner and evaluate
res_spr <- resample(hotels_task, super_lrn, cv5, store_models = TRUE)
res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.tpr"),
                       msr("classif.fnr"),
                       msr("classif.tnr"),
                       msr("classif.auc"),
                       msr("classif.logloss")))
