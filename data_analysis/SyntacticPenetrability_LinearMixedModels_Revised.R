# CRITICAL ANALYSIS LOOPED
library(lmerTest)
setwd("/Path/To/Llama2_critical_and_lure_analyses")
Llama2_combined <- read.csv("Llama2_combined.csv")
pooled <- Llama2_combined

pooled$sentence_pair <- as.factor(pooled$sentence_pair)

pairs_subset <- unique(pooled[, c("dependency", "head_number")])
print(pairs_subset)


pvals <- numeric(0)
print(pvals)

for (i in seq_len(nrow(pairs_subset))) {
  dep_i  <- pairs_subset$dependency[i]
  head_i <- pairs_subset$head_number[i]
  
  data <- subset(pooled, dependency == dep_i & head_number == head_i)
  data$sentence_pair <- as.factor(data$sentence_pair)
  
  data$Logit_Transform <- log(data$Attention_Strength / (1 - data$Attention_Strength))
  
  cat("\n==============================\n",
      "Dependency:", dep_i, "| Head:", head_i, "\n",
      "N rows:", nrow(data), "\n",
      "==============================\n")
  
  ### MIXED EFFECTS MODEL
  Critical_model <- lmer(Logit_Transform ~ Plausibility + LgSUBTLWF + PMI_crit + Cosine_crit + (1 | sentence_pair), data = data)
  print(summary(Critical_model))
  pvals <- c(pvals, coef(summary(Critical_model))["PlausibilityPlausible", "Pr(>|t|)"])
}

names(pvals) <- paste(pairs_subset$dependency, pairs_subset$head_number, sep = "__")
pvec_crit <- pvals[is.finite(pvals)]  

fdrs <- p.adjust(pvec_crit, method = "BH")
print(fdrs)

significant_fdrs <- fdrs[fdrs < 0.05]
print(significant_fdrs)   




###########################################################################
# LURE ANALYSIS LOOPED

library(lmerTest)

setwd("/Path/To/Llama2_critical_and_lure_analyses")
Llama2_combined <- read.csv("Llama2_combined.csv")
pooled <- Llama2_combined

pooled$sentence_pair <- as.factor(pooled$sentence_pair)

pairs <- unique(pooled[, c("dependency", "head_number")])
print(pairs)
pairs_subset <- head(pairs, 15)    ## Subset to exclude rows where there is no lure data (e.g., for pobj in decoder-only models)
print(pairs_subset)


pvals <- numeric(0)
print(pvals)

for (i in seq_len(nrow(pairs_subset))) {
  dep_i  <- pairs_subset$dependency[i]
  head_i <- pairs_subset$head_number[i]
  
  data <- subset(pooled, dependency == dep_i & head_number == head_i)
  data$sentence_pair <- as.factor(data$sentence_pair)
  
  data$Lure_Logit_Transform <- log(data$Lure_attention_strength / (1 - data$Lure_attention_strength))
  
  # Simple header for readability
  cat("\n==============================\n",
      "Dependency:", dep_i, "| Head:", head_i, "\n",
      "N rows:", nrow(data), "\n",
      "==============================\n")
  
  ### MIXED EFFECTS MODEL
  Lure_model <- lmer(Lure_Logit_Transform ~ Plausibility + Lure_LgSUBTLWF + PMI_lure + Cosine_lure + (1 | sentence_pair), data = data)
  print(summary(Lure_model))
  pvals <- c(pvals, coef(summary(Lure_model))["PlausibilityPlausible", "Pr(>|t|)"])
}

names(pvals) <- paste(pairs_subset$dependency, pairs_subset$head_number, sep = "__")
pvec_lure <- pvals[is.finite(pvals)] 
print(pvec_lure)
fdrs <- p.adjust(pvec_lure, method = "BH")
print(fdrs)

significant_fdrs_lure <- fdrs[fdrs < 0.05]
print(significant_fdrs_lure)  


###########################################################################
###########################################################################
#Lure x baseline (control) interaction analysis - all models

library(lmerTest)
library(emmeans)

setwd("/Path/to/All_models_lure_specificity_analysis")

BERT_nsubjpass_lure_control <- read.csv("BERT_nsubjpass_lure_control.csv")
GPT2_nsubj_lure_control <- read.csv("GPT2_nsubj_lure_control.csv")
GPT2_nsubjpass_lure_control <- read.csv("GPT2_nsubjpass_lure_control.csv")
Llama2_nsubj_lure_control <- read.csv("Llama2_nsubj_lure_control.csv")
Llama2_nsubjpass_lure_control <- read.csv("Llama2_nsubjpass_lure_control.csv")

###########################################################################

data <- BERT_nsubjpass_lure_control
data$sentence_pair <- as.factor(data$sentence_pair)

data$Logit_Transform <- log(data$Lure_and_control_attn / (1 - data$Lure_and_control_attn))

### MIXED EFFECTS MODEL
lure_control_interaction_model <- lmer(Logit_Transform ~ Plausibility*lure_or_control + Lure_or_Control_LgSUBTLWF + (1 | sentence_pair), data = data)
summary(lure_control_interaction_model)

### Fixed Effects Model
lure_control_interaction_model <- lm(Logit_Transform ~ Plausibility*lure_or_control + Lure_or_Control_LgSUBTLWF, data = data)
summary(lure_control_interaction_model)

emm_model <- emmeans(lure_control_interaction_model, ~ Plausibility | lure_or_control)
# Perform pairwise comparisons within each level of Lure_or_Control
contrast(emm_model, method = "pairwise", adjust = "bonferroni")

