library(bbsBayes2)
library(sf)
library(dplyr)
library(ggplot2)
library(parallel)
library(jsonlite)  


config <- fromJSON("config.json")

fetch_bbs_data()

setwd(config$working_directory)

map <- sf::read_sf(config$shapefile_path)
map$STRAT <- c(1,1)
map <- rename(map, strata_name = layer)

bird_names <- read.csv(config$bird_names_file)

process_species <- function(species) {
  tryCatch({
    cat(sprintf("Processing species: %s\n", species))
    
    s <- stratify(by = "GYE", species = species, strata_custom = map)
    p <- prepare_data(s, min_year = config$min_year, max_year = config$max_year)
    pm <- prepare_model(p, model = "gamye", model_variant = "hier", calculate_cv = TRUE)
    m <- run_model(pm, refresh = 10, iter_warmup = 1000, iter_sampling = 2000, adapt_delta = 0.8, max_treedepth = 15)
    
    i <- generate_indices(m)
    i2 <- as.data.frame(i$indices)
    i2$species <- species
    
    i2$significant_90 <- ifelse(i2$index_q_0.05 > 0 & i2$index_q_0.95 > 0, "Positive",
                                ifelse(i2$index_q_0.05 < 0 & i2$index_q_0.95 < 0, "Negative", "Not Significant"))
    
    significance_summary <- data.frame(
      year = i2$year,
      significant = i2$significant_90,
      lower_bound = i2$index_q_0.05,
      upper_bound = i2$index_q_0.95
    )
    
    significance_summary$lower_diff <- c(NA, diff(significance_summary$lower_bound))
    significance_summary$upper_diff <- c(NA, diff(significance_summary$upper_bound))
    
    significance_summary$year_significance <- ifelse(
      significance_summary$lower_bound > 0 & significance_summary$upper_bound > 0, "Positive",
      ifelse(significance_summary$lower_bound < 0 & significance_summary$upper_bound < 0, "Negative", "Not Significant")
    )
    
    filename <- sprintf("%s%s.csv", config$output_indices_folder, species)
    write.csv(i2, filename)
    
    t <- generate_trends(i)
    t2 <- as.data.frame(t$trends)
    t2$species <- species
    filename <- sprintf("%s%s_Data.csv", config$output_trends_folder, species)
    write.csv(t2, filename)
    
    cat(sprintf("Finished processing species: %s\n", species))
    
  }, error = function(e) {
    message(sprintf("Error processing species %s: %s", species, e$message))
  })
}

num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
clusterExport(cl, list("process_species", "map", "stratify", "prepare_data", "prepare_model", "run_model", "generate_indices", "generate_trends", "write.csv", "bird_names"))

parLapply(cl, bird_names$species, process_species)

stopCluster(cl)

folder_path <- config$output_trends_folder
csv_files <- list.files(path = folder_path, pattern = "*.csv", full.names = TRUE)
combined_data <- csv_files %>%
  lapply(read.csv) %>%
  bind_rows()
write.csv(combined_data, file = config$combined_trends_file, row.names = FALSE)
cat(sprintf("All CSV files have been combined and saved as '%s'.\n", config$combined_trends_file))

folder_path <- config$output_indices_folder
csv_files <- list.files(path = folder_path, pattern = "*.csv", full.names = TRUE)
combined_data <- csv_files %>%
  lapply(read.csv) %>%
  bind_rows()
write.csv(combined_data, file = config$combined_indices_file, row.names = FALSE)
cat(sprintf("All CSV files have been combined and saved as '%s'.\n", config$combined_indices_file))