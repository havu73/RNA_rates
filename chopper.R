args <- commandArgs(trailingOnly=TRUE)

dir <- args[1]
refbed <- args[2]
dir_out <- args[3]
parameters <- args[4]

# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
#
# #BiocManager::install("ChIPsim")
# BiocManager::install("Biostrings")
#
# library(ChIPsim) # To save as fastq
library(Biostrings)

library(tidyverse) # in cluster R
library(stringr) # in cluster R
library(plyr) # in cluster R
library(tidyr) # in cluster R
library(stringi) # in cluster R


#This script chops long reads to create PAIRED END short reads
#I recommend cleaning the simulated reads a bit by doing:
#for i in *.csv.gz;do gunzip -cd $i| sed 's|[[,]||g' | sed 's|[],]||g'|gzip > $i.mod.gz;done



#Chopper
get_reads <- function(lengths, eta_val = 200, insertsize = c(200, 300)){
  # Select a fragment from each transcript, size select, and return the starting position of the
  # resulting reads relative to the length of the transcript
  # Inputs: 
  #   lengths - the lengths of the transcript
  #   eta_val - the eta value input to the Weibull distribution
  #   insertsize - a list of length two with the lower and upper bound for fragments
  # Outputs:
  #   fragments - a data frame with the start positions of the filtered fragments and the index
  #               of the transcript that it came from (columns: transcript, start)
  
  # sample lengths from a Weibull distribution for each transcript and transform them to the length 
  # of the transcripts
  deltas = log10(lengths)
  ns_minus_1 = pmax(round(lengths/eta_val/gamma(1/deltas + 1)) - 1, 0)
  xis = lapply(ns_minus_1, function(n) {diff(sort(c(runif(n), 0, 1)))})
  xis_transformed = mapply(function(x, d) {x^(1/d)}, xis, deltas, SIMPLIFY = F)
  delta_is = mapply(function(len, x_t) {round(len*x_t/sum(x_t))}, lengths, xis_transformed, SIMPLIFY = F)

# get all the start and end points of the fragments 
  starts = lapply(delta_is, function(d) {
    if (length(d) > 1) {
      c(sample(min(insertsize[1], d[1]), 1), cumsum(d[1:(length(d)-1)]))
    } else{
      sample(min(insertsize[1], d), 1)
    }
  })
  ends = lapply(delta_is, function(d) {
    if (length(d) > 1) {
      c(cumsum(d[1:(length(d)-1)]), sum(d)-sample(min(insertsize[1], sum(d) - d[length(d)]), 1))
    } else{
      d
    }
  })

  # convert to a data frame of fragments and associated transcript index
  #fragments = data.frame(transcript = rep(1:length(deltas), lengths(delta_is)),
  fragments = data.frame(transcript = rep(1:length(deltas), unlist(lapply(delta_is, length))),
                       start = unlist(starts),
                       end = unlist(ends))
  fragments$length = fragments$end - fragments$start

  # Filter fragments by length and return
  fragments = fragments[fragments$length >= insertsize[1] & fragments$length <= insertsize[2],]
  return(fragments[c('transcript', 'start','end')])
}
#Read creator
read_maker_namer <- function(current_coordinate,read_function,strand,readlength=150){#Assumes paired end seq 150ea
  short_reads_R1_function <- list()
  short_reads_R2_function <- list()
  temp_merged_list <- list()
  new_short_read<-  substr(read_function, start = current_coordinate['start'], stop = as.numeric(current_coordinate['end'])+readlength*2-1)
  
  insert_length <- nchar(new_short_read) #Get insert length
  per_pair_length<- read_length/2 #Get read length for each pair in case it's paired seq  --> this is actually a bug because read_length is not defined in the function
  
  #The sequence I'm getting are for mRNA and coded in the right way. Now I need to switch strand assuming LP is fr-firststrand
  #Reverse the mRNA seq for R1
  
  R1 <- substr(new_short_read, start = insert_length-readlength+1, stop = read_length)
  R2 <- substr(new_short_read, start = 1, stop = readlength)
  
  
  #if (strand=='+'){
  #  R1 <- substr(new_short_read, start = insert_length-readlength+1, stop = read_length)
  #  R2 <- substr(new_short_read, start = 1, stop = readlength)} # - strand genes are already revcomp, so this is valid for both gene types
  
  #if (strand=='-'){
  #  R1 <- substr(new_short_read, start = 1, stop = readlength)
  #  R2 <- substr(new_short_read, start = insert_length-readlength+1, stop = read_length)} # - strand genes are already revcomp, so this is valid for both gene types
  
  R1 <- as.character(Biostrings::reverseComplement(DNAString(R1)))
  
  
  read_name_temp <- paste(sep = '',stri_rand_strings(1, 20, pattern = "[A-Za-z0-9]"),'_',current_coordinate['counts_5eU_subs'],':U>U','-',current_coordinate['counts_6sG_subs'],':G>A','-',current_coordinate['counts_4sU_subs'],':T>C','_',current_coordinate['time']) #Create a read name with subs counts
  
  
  read_name_temp <- gsub(" ", "", read_name_temp)


  read_qual_each <- stri_rand_strings(1, readlength, pattern = "[A-I]") #Generate quality score for both reads

  short_reads_temp_R1 <- list(paste('@',read_name_temp,sep=''),R1,paste('+',read_name_temp,sep=''),read_qual_each)
  short_reads_temp_R2 <- list(paste('@',read_name_temp,sep=''),R2,paste('+',read_name_temp,sep=''),read_qual_each)

  short_reads_R1_function <- append(short_reads_R1_function,short_reads_temp_R1)
  short_reads_R2_function <- append(short_reads_R2_function,short_reads_temp_R2)

  temp_merged_list <- list('short_reads_R1_function'=short_reads_R1_function,'short_reads_R2_function'=short_reads_R2_function)

  return(temp_merged_list)
}

# Rscript chopper.R ./1_mRNA_generator  Homo_sapiens.GRCh38.95.uniquegene.bed ./2_chopper parameters
gtf <- read.table(refbed)
#gtf <- read.table('/pi/athma.pai-umw/analyses/jesse/KB/current_sci/essential_simulation_files/Homo_sapiens.GRCh38.95.uniquegene.bed')
colnames(gtf) <- c('chr','start','end','gene_id','score','strand')

# Loop through each parameter directory within the parent directory, and run the following loop for each simulated gene .csv file
#parent_directory <- "/pi/athma.pai-umw/analyses/jesse/KB/current_sci/simulations/reads_200/bysubrate"
#directories <- list.dirs(parent_directory, recursive=F)
#filt_directories <- directories[str_detect(directories, "5eUrates_")]

#for (dir in filt_directories[1]) {
  #setwd('dir')
  #Use parent directory within which all substitution rate directories with .csv files are.
  all_files <- list.files(path = dir, pattern = "\\.tsv$", full.names = TRUE)

  #set.seed(758)
  #alpha <- 0.8

  value_counts <- 1
  for (file in all_files){
    print(paste(value_counts,'/',length(all_files),' genes ran',sep='')) #Print iteration
    value_counts <- value_counts+1 #Update iteration
    
    test <- read.table(file = file, sep = '\t', header = F, stringsAsFactors = FALSE)
    gene <- gsub("_.*", "", file)
    strand <- gtf[gtf$V4 == gene, 5][1]
    colnames(test)[c(7,9,16,12,13,14)] <- c('starts_6sG_sites','starts_4sU_sites','read','position_U2U_conversions','position_G2A_conversions','position_U2C_conversions')
    
    for (read_size in c(300)){ #Sequencing cycle number (divide by 2 to get read size)
      #The input to the chopper is a vector of transcript lengths
      read_name <- c()
      read_qual_all <- c()
      short_reads_all <- list()
      
      for (i in 1:nrow(test)){
        read <- test$read[i]
        read_length <- read %>% nchar() #Get read length
        coordinates_to_chop <- get_reads(read_length)
        #Get subtitution positions
        subs_coordinates_5eU <- str_extract_all(test$position_U2U_conversions[i], "[\\.0-9e-]+") %>% unlist() %>% as.numeric()
        subs_coordinates_6sG <- str_extract_all(test$position_G2A_conversions[i], "[\\.0-9e-]+") %>% unlist() %>% as.numeric()
        subs_coordinates_4sU <- str_extract_all(test$position_U2C_conversions[i], "[\\.0-9e-]+") %>% unlist() %>% as.numeric()
        if (nrow(coordinates_to_chop)>0){
            coordinates_to_chop$counts_5eU_subs <- 0
            coordinates_to_chop$counts_6sG_subs <- 0
            coordinates_to_chop$counts_4sU_subs <- 0
          
            matrix_5eU_temp <- apply(coordinates_to_chop,1,function(x) subs_coordinates_5eU>=x[2] & subs_coordinates_5eU<=x[3]) %>% which(arr.ind = T)
            matrix_6sG_temp <- apply(coordinates_to_chop,1,function(x) subs_coordinates_6sG>=x[2] & subs_coordinates_6sG<=x[3]) %>% which(arr.ind = T)
            matrix_4sU_temp <- apply(coordinates_to_chop,1,function(x) subs_coordinates_4sU>=x[2] & subs_coordinates_4sU<=x[3]) %>% which(arr.ind = T)

            #Get substitutions counts per read
            if (class(matrix_5eU_temp)[1]!='integer'){#Only do it for labeled transcripts
            matrix_5eU_temp <- matrix_5eU_temp[,2] %>% table() %>% as.data.frame()
            rows_5eU <- as.numeric(as.character(matrix_5eU_temp$.))
            coordinates_to_chop[rows_5eU,'counts_5eU_subs'] <- matrix_5eU_temp$Freq}
            if (class(matrix_6sG_temp)[1]!='integer'){#Only do it for labeled transcripts
            matrix_6sG_temp <- matrix_6sG_temp[,2] %>% table() %>% as.data.frame()
            rows_6sG <- as.numeric(as.character(matrix_6sG_temp$.))
            coordinates_to_chop[rows_6sG,'counts_6sG_subs'] <- matrix_6sG_temp$Freq}
            if (class(matrix_4sU_temp)[1]!='integer'){#Only do it for labeled transcripts
            matrix_4sU_temp <- matrix_4sU_temp[,2] %>% table() %>% as.data.frame()
            rows_4sU <- as.numeric(as.character(matrix_4sU_temp$.))
            coordinates_to_chop[rows_4sU,'counts_4sU_subs'] <- matrix_4sU_temp$Freq}

          #pulldown module, modeling pulldown efficiency and noise
          #with_5eU <- subset(coordinates_to_chop, counts_5eU_subs > 0)
          #without_5eU <- subset(coordinates_to_chop, counts_5eU_subs == 0)

          #pulldown_freq <- round(alpha*(nrow(with_5eU)))
          #noise_freq <- round((1-alpha)*(nrow(without_5eU)))

          #pulldown <- with_5eU[sample(nrow(with_5eU), pulldown_freq, replace = F), ]
          #noise <- without_5eU[sample(nrow(without_5eU), noise_freq, replace = F), ]

          #if (nrow(pulldown)>0 | nrow(noise)>0){
          #coordinates_to_chop <- rbind(pulldown, noise)}

          #Get the position at each time limit
          min5_limit <- test$starts_6sG_sites[i]
          min10_limit <- test$starts_4sU_sites[i]
          coordinates_to_chop$time <- 'NA'
          coordinates_to_chop$time[coordinates_to_chop$start<min5_limit] <- "0-5min"
          coordinates_to_chop$time[coordinates_to_chop$start>=min5_limit] <- "5-10min"
          coordinates_to_chop$time[coordinates_to_chop$start>=min10_limit] <- "10-15min"
          reads_one_iteration <- apply(coordinates_to_chop,1,read_maker_namer,read,strand)  # Get reads for each transcript, number of reads is equal to # rows in coordinates_to_chop
          splitted_reads <- do.call(mapply, c(list, reads_one_iteration, SIMPLIFY=FALSE)) #Split output (one per read1/2)

          short_reads_all <- append(short_reads_all,splitted_reads)}
        short_reads_all <- sapply(unique(names(short_reads_all)), function(x) unname(unlist(short_reads_all[names(short_reads_all)==x])), simplify=FALSE) #Merge all R1 and all R2
      }

      name_for_fastq_R1 <- file.path(dir_out,paste0(parameters,"_",gene,"_",read_size/2,'PE','_R1.fastq.gz'))
      name_for_fastq_R2 <- str_replace(name_for_fastq_R1,'_R1.','_R2.')

      short_reads_R1 <- short_reads_all["short_reads_R1_function"] %>% unlist()
      short_reads_R2 <- short_reads_all["short_reads_R2_function"] %>% unlist()

      gz1 <- gzfile(name_for_fastq_R1, "w")
      write.table(short_reads_R1, gz1,quote = F,row.names = F,col.names = F)
      close(gz1)

      gz2 <- gzfile(name_for_fastq_R2, "w")
      write.table(short_reads_R2, gz2,quote = F,row.names = F,col.names = F)
      close(gz2)
    }
  }
#}
# close directory loop


