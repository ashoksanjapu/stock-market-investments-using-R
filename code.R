library(readxl)
library(ggplot2)
library(dplyr)
library(tidyverse)
att_data <- read_excel("/Users/ashoksanjapu/Downloads/Data Analytics1_ sushma/Project_ADTA_DataSet.xlsx", sheet = "T")

# 1. Time Series Plot for 'Adj Close' Price
#ggplot(att_data, aes(x = Date, y = `Adj Close`)) +
#  geom_line() +
#  labs(title = "Adjusted Close Price Over Time for AT&T",
#      x = "Date",
#      y = "Adjusted Close Price") +
# theme_minimal()

# 2. Volume-Price Trend Chart
#att_data %>%
# ggplot(aes(x = Volume, y = `Adj Close`)) +
# geom_point(alpha = 0.5) +
# geom_smooth(method = "lm", color = "blue") +
# labs(title = "Volume vs. Adjusted Close Price for AT&T",
#     x = "Volume",
#      y = "Adjusted Close Price") +
# theme_minimal()



library(readxl)
library(ggplot2)
library(dplyr)



att_data <- read_excel("/Users/ashoksanjapu/Downloads/Data Analytics1_ sushma/Project_ADTA_DataSet.xlsx", sheet = "T") %>% mutate(Company = "AT&T")
nvidia_data <- read_excel("/Users/ashoksanjapu/Downloads/Data Analytics1_ sushma/Project_ADTA_DataSet.xlsx", sheet = "NVDA") %>% mutate(Company = "NVIDIA")
JPMorgan_data <- read_excel("/Users/ashoksanjapu/Downloads/Data Analytics1_ sushma/Project_ADTA_DataSet.xlsx", sheet = "JPMCL.SN") %>% mutate(Company = "JP Morgan")
CMG_data <- read_excel("/Users/ashoksanjapu/Downloads/Data Analytics1_ sushma/Project_ADTA_DataSet.xlsx", sheet = "CMG") %>% mutate(Company = "CMG")
SPOT_data <- read_excel("/Users/ashoksanjapu/Downloads/Data Analytics1_ sushma/Project_ADTA_DataSet.xlsx", sheet = "SPOT") %>% mutate(Company = "SPOT")
TATAELXSI_data <- read_excel("/Users/ashoksanjapu/Downloads/Data Analytics1_ sushma/Project_ADTA_DataSet.xlsx", sheet = "TATAELXSI.NS") %>% mutate(Company = "TATAELXSI")
META_data <- read_excel("/Users/ashoksanjapu/Downloads/Data Analytics1_ sushma/Project_ADTA_DataSet.xlsx", sheet = "META") %>% mutate(Company = "META")




convert_to_numeric <- function(x) {
  # First, convert any factors to character if they are not already
  x <- as.character(x)
  
  # Remove commas, dollar signs, and other non-numeric characters
  x <- gsub("[^0-9.-]", "", x)
  
  # Convert to numeric
  as.numeric(x)
}


# Apply the conversion to the 'Open' column for each dataset
att_data$Open <- convert_to_numeric(att_data$Open)
CMG_data$Open <- convert_to_numeric(CMG_data$Open)
JPMorgan_data$Open <- convert_to_numeric(JPMorgan_data$Open)
META_data$Open <- convert_to_numeric(META_data$Open)
nvidia_data$Open <- convert_to_numeric(nvidia_data$Open)
SPOT_data$Open <- convert_to_numeric(SPOT_data$Open)
TATAELXSI_data$Open <- convert_to_numeric(TATAELXSI_data$Open)

# Convert 'High' column for all datasets to numeric
att_data$High <- convert_to_numeric(att_data$High)
CMG_data$High <- convert_to_numeric(CMG_data$High)
JPMorgan_data$High <- convert_to_numeric(JPMorgan_data$High)
META_data$High <- convert_to_numeric(META_data$High)
nvidia_data$High <- convert_to_numeric(nvidia_data$High)
SPOT_data$High <- convert_to_numeric(SPOT_data$High)
TATAELXSI_data$High <- convert_to_numeric(TATAELXSI_data$High)

# Convert 'Low' column for all datasets to numeric
att_data$Low <- convert_to_numeric(att_data$Low)
CMG_data$Low <- convert_to_numeric(CMG_data$Low)
JPMorgan_data$Low <- convert_to_numeric(JPMorgan_data$Low)
META_data$Low <- convert_to_numeric(META_data$Low)
nvidia_data$Low <- convert_to_numeric(nvidia_data$Low)
SPOT_data$Low <- convert_to_numeric(SPOT_data$Low)
TATAELXSI_data$Low <- convert_to_numeric(TATAELXSI_data$Low)

# Convert 'Close' column for all datasets to numeric
att_data$Close <- convert_to_numeric(att_data$Close)
CMG_data$Close <- convert_to_numeric(CMG_data$Close)
JPMorgan_data$Close <- convert_to_numeric(JPMorgan_data$Close)
META_data$Close <- convert_to_numeric(META_data$Close)
nvidia_data$Close <- convert_to_numeric(nvidia_data$Close)
SPOT_data$Close <- convert_to_numeric(SPOT_data$Close)
TATAELXSI_data$Close <- convert_to_numeric(TATAELXSI_data$Close)


# Convert 'Adj Close' column for all datasets to numeric
att_data$AdjClose <- convert_to_numeric(att_data$AdjClose)
CMG_data$AdjClose <- convert_to_numeric(CMG_data$AdjClose)
JPMorgan_data$AdjClose <- convert_to_numeric(JPMorgan_data$AdjClose)
META_data$AdjClose <- convert_to_numeric(META_data$AdjClose)
nvidia_data$AdjClose <- convert_to_numeric(nvidia_data$AdjClose)
SPOT_data$AdjClose <- convert_to_numeric(SPOT_data$AdjClose)
TATAELXSI_data$AdjClose <- convert_to_numeric(TATAELXSI_data$AdjClose)


# Convert 'Volume' column for all datasets to numeric
att_data$Volume <- convert_to_numeric(att_data$Volume)
CMG_data$Volume <- convert_to_numeric(CMG_data$Volume)
JPMorgan_data$Volume <- convert_to_numeric(JPMorgan_data$Volume)
META_data$Volume <- convert_to_numeric(META_data$Volume)
nvidia_data$Volume <- convert_to_numeric(nvidia_data$Volume)
SPOT_data$Volume <- convert_to_numeric(SPOT_data$Volume)
TATAELXSI_data$Volume <- convert_to_numeric(TATAELXSI_data$Volume)



install.packages("tidyverse")

# Load tidyverse to use its functions
library(tidyverse)

# Now you should be able to use drop_na() without errors

att_data <- drop_na(att_data)
CMG_data <- drop_na(CMG_data)
JPMorgan_data <- drop_na(JPMorgan_data)
META_data <- drop_na(META_data)
nvidia_data <- drop_na(nvidia_data)
SPOT_data <- drop_na(SPOT_data)
TATAELXSI_data <- drop_na(TATAELXSI_data)

combined_data <- bind_rows(att_data, CMG_data, JPMorgan_data, META_data, nvidia_data, SPOT_data, TATAELXSI_data)

# Plot all companies on a single graph with facets and different colors for each company's line
ggplot(combined_data, aes(x = Date, y = `AdjClose`, group = Company, color = Company)) +
  geom_line() +
  facet_wrap(~ Company, scales = "free_y") + # Allows for separate y-axes
  labs(title = "Adjusted Close Price Over Time by Company",
       x = "Date",
       y = "Adjusted Close Price") +
  theme_minimal() +
  theme(legend.position = "none")

#------------chart2--------


# Calculate daily returns for each company
combined_data <- combined_data %>%
  arrange(Company, Date) %>%
  group_by(Company) %>%
  mutate(Daily_Return = (AdjClose / lag(AdjClose) - 1) * 100) %>%
  ungroup()

# Remove the first NA value of each company's daily returns
combined_data <- combined_data %>% 
  group_by(Company) %>%
  filter(!is.na(Daily_Return)) %>%
  ungroup()

# Create a boxplot for daily returns
ggplot(combined_data, aes(x = Company, y = Daily_Return, fill = Company)) +
  geom_boxplot() +
  labs(title = "Distribution of Daily Returns by Company",
       x = "Company",
       y = "Daily Return (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 


#--------- chart3 --------------

install.packages("reshape2")

library(reshape2)
library(ggplot2)

# Assuming combined_data is your combined dataset and it's ordered by Date within each Company

# Calculate daily returns for each company
combined_data <- combined_data %>%
  arrange(Company, Date) %>%
  group_by(Company) %>%
  mutate(Daily_Return = (AdjClose / lag(AdjClose) - 1)) %>%
  ungroup()

# Drop rows with NA (which will be the first row of each company due to the lag)
combined_data <- drop_na(combined_data, Daily_Return)

# Reshape data for correlation calculation
returns_wide <- combined_data %>%
  select(Date, Company, Daily_Return) %>%
  spread(Company, Daily_Return)

# Calculate correlations
correlations <- cor(returns_wide[,-1], use = "complete.obs")  # Exclude Date column for correlation

# Melt the correlation matrix for ggplot
correlation_melted <- melt(correlations)

# Plot the heatmap
ggplot(correlation_melted, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  labs(x = '', y = '', title = 'Correlation Heatmap of Daily Returns')


#---------------Final part------------


str(combined_data)
summary(combined_data)

# Show count of missing values before handling
sum(is.na(combined_data))

# Remove rows with missing values
data_clean <- combined_data %>% drop_na()

# Show count of missing values after handling
sum(is.na(data_clean))

# Check for duplicates and remove them
data_unique <- distinct(data_clean)

print(data_unique)



#--------- Normalization----------

# Normalize and standardize numerical data
data_standardized <- data_unique %>% 
  mutate(across(where(is.numeric), scale))  # Z-score standardization

data_normalized <- data_unique %>% 
  mutate(across(where(is.numeric), ~(. - min(.)) / (max(.) - min(.)))) 

print(data_normalized)

print(data_standardized)
options(scipen = 999)

#---- Time series plot-------

ggplot(combined_data, aes(x = Date, y = Close, color = Company)) +
  geom_line() +
  labs(title = "Time Series of Closing Prices", x = "Date", y = "Closing Price") +
  theme_minimal()


#----- 

install.packages("quantmod")

library(quantmod)

getSymbols("T", src = "yahoo", from = "2020-01-01", to = "2023-01-01")
chartSeries(T, TA=NULL, theme="white", up.col='green', dn.col='red', type='candles')

##plot 2

ggplot(combined_data, aes(x = Date, y = Volume, fill = Company)) +
  geom_bar(stat = "identity") +
  labs(title = "Volume Traded Over Time", x = "Date", y = "Volume") +
  theme_minimal()


##plot 3

ggplot(combined_data, aes(x = Daily_Return, fill = Company)) +
  geom_histogram(bins = 30, alpha = 0.7) +
  labs(title = "Distribution of Daily Returns", x = "Daily Return", y = "Frequency") +
  theme_minimal()

##plot 4

average_returns <- combined_data %>%
  group_by(Company) %>%
  summarise(Average_Daily_Return = mean(Daily_Return, na.rm = TRUE))

ggplot(average_returns, aes(x = Company, y = Average_Daily_Return, fill = Company)) +
  geom_col() +
  labs(title = "Average Daily Return by Company",
       x = "Company", y = "Average Daily Return (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

##plot 5

library(ggridges)

combined_data$Year <- as.numeric(format(as.Date(combined_data$Date), "%Y"))

ggplot(combined_data, aes(x = Daily_Return, y = factor(Year), fill = Year)) +
  geom_density_ridges_gradient(scale = 3, rel_min_height = 0.01) +
  labs(title = "Distribution of Daily Returns Over Years",
       x = "Daily Return", y = "Year") +
  theme_ridges() +
  scale_fill_viridis_c()

# Compute the summary statistics
summary_stats <- combined_data %>%
  select_if(is.numeric) %>%  # Selecting only numeric columns
  summarise_all(list(
    mean = ~mean(., na.rm = TRUE),
    median = ~median(., na.rm = TRUE),
    sd = ~sd(., na.rm = TRUE),
    IQR = ~IQR(., na.rm = TRUE)
  ))



# Transpose the summary statistics for better readability
summary_stats_transposed <- t(summary_stats)

# Convert to a data frame for better formatting, with variable names as the first column
summary_stats_df <- as.data.frame(summary_stats_transposed)

# Print the transposed summary statistics
print(summary_stats_df)

#------------ box plot --------
combined_data %>%
  select_if(is.numeric) %>%
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  facet_wrap(~variable, scales = "free_x") +
  theme_minimal() +
  labs(title = "Distribution of Numeric Variables", x = "Value", y = "Frequency")


combined_data %>%
  select_if(is.numeric) %>%
  gather(key = "variable", value = "value") %>%
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(fill = "lightblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Boxplot of Numeric Variables", x = "Variable", y = "Value")



#--------Modeling --------------


# Convert Date to Date format if necessary
combined_data$Date <- as.Date(combined_data$Date, format = "%Y-%m-%d")

# Select relevant columns for the analysis
model_data <- combined_data %>% select(Daily_Return, Open, High, Low, Close, Volume)

# Check for NA values and remove if necessary
model_data <- na.omit(model_data)


library(ggplot2)

# Plotting Daily Return vs Volume
ggplot(model_data, aes(x = Volume, y = Daily_Return)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", color = "blue", se = FALSE) +
  labs(title = "Daily Return vs Volume", x = "Volume", y = "Daily Return")

# Building a linear model
lm_model <- lm(Daily_Return ~ Open + High + Low + Close + Volume, data = model_data)
summary(lm_model)

# Plotting residuals to check assumptions
par(mfrow=c(2,2))
plot(lm_model)


# Checking residuals
plot(lm_model$residuals, main="Residuals Plot", ylab="Residuals", xlab="Fitted values")
abline(h=0, col="red")


# Building a more complex model with interaction terms and polynomial terms
lm_model2 <- lm(Daily_Return ~ poly(Open, 2) + poly(High, 2) + poly(Low, 2) + poly(Close, 2) + Volume + I(Open*Close) + I(High*Low), data = model_data)
summary(lm_model2)

# Plotting residuals to check assumptions for the new model
par(mfrow=c(2,2))
plot(lm_model2)

# Checking residuals
plot(lm_model2$residuals, main="Residuals Plot", ylab="Residuals", xlab="Fitted values")
abline(h=0, col="red")



#--------regressionmodel---------------


library(randomForest)

# Ensure that the data does not have any missing values
model_data <- na.omit(model_data)

# Building a random forest model
rf_model <- randomForest(Daily_Return ~ Open + High + Low + Close + Volume, data=model_data, ntree=500, mtry=3, importance=TRUE)

# Summarize the model
print(summary(rf_model))

# Check variable importance
importance(rf_model)

# Plotting model importance
varImpPlot(rf_model)


library(ggplot2)

# Coefficient plot
coef_df <- as.data.frame(summary(lm_model)$coefficients)
reset <- rownames(coef_df)
coef_df <- cbind(Variable = reset, coef_df)
coef_df$Variable <- as.character(coef_df$Variable)

# Correcting the standard error column name
ggplot(coef_df, aes(x=Variable, y=Estimate, fill=Variable)) +
  geom_col() +
  geom_errorbar(aes(ymin=Estimate-`Std. Error`, ymax=Estimate+`Std. Error`), width=.2) +
  labs(title="Coefficient Plot of Linear Model", x="Variables", y="Coefficients")

#---------anova------

combined_data$Company <- as.factor(combined_data$Company)
combined_data$Daily_Return <- as.numeric(combined_data$Daily_Return)

# Performing ANOVA
anova_results <- aov(Daily_Return ~ Company, data = combined_data)
summary(anova_results)

# To get a more detailed view of the differences between groups
library(emmeans)
pairwise <- emmeans(anova_results, pairwise ~ Company)
summary(pairwise$contrasts)

install.packages("caret")
library(caret)

# Convert Daily_Return to a binary outcome
combined_data$Return_Binary <- ifelse(combined_data$Daily_Return > 0, 1, 0)

# Splitting the data into training and test sets
set.seed(123)
training_indices <- createDataPartition(combined_data$Return_Binary, p = 0.8, list = FALSE)
training_data <- combined_data[training_indices, ]
test_data <- combined_data[-training_indices, ]

# Convert Daily_Return to a binary outcome
combined_data$Return_Binary <- ifelse(combined_data$Daily_Return > 0, 1, 0)

# Splitting the data into training and test sets
set.seed(123)
training_indices <- createDataPartition(combined_data$Return_Binary, p = 0.8, list = FALSE)
training_data <- combined_data[training_indices, ]
test_data <- combined_data[-training_indices, ]

# Building the logistic regression model
logistic_model <- glm(Return_Binary ~ Open + High + Low + Close + Volume, data = training_data, family = binomial())

# Making predictions on the test set
test_data$predicted_class <- ifelse(predict(logistic_model, newdata = test_data, type = "response") > 0.5, 1, 0)

# Calculating precision, recall, and F1 score
conf_matrix <- table(test_data$predicted_class, test_data$Return_Binary)
precision <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
recall <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print results
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1_score))

