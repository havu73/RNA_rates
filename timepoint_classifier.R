# Notes for caller:
# - make directory with all pdf/png output files in the reads_200 parent directory, call it timepoint_classification. Only need it once, outside any loop

#args <- commandArgs(trailingOnly=TRUE)

#parent_dir <- args[1]
#final_dir <- args[2]

parent_dir <- '/pi/athma.pai-umw/analyses/jesse/KB/current_sci/simulations/reads_200/bysubrate'
final_dir <- '/pi/athma.pai-umw/analyses/jesse/KB/current_sci/simulations/reads_200/bysubrate' #/timepoint_classification'
#ref <- '/pi/athma.pai-umw/genomes/hg38/Homo_sapiens.GRCh38.95.uniquegene.bed.bed'

#setwd(parent_dir)
setwd('/pi/athma.pai-umw/analyses/jesse/KB/current_sci/simulations/reads_200/bysubrate') #/5eUrates_5-5_6sGrates_5-5_4sUrates_5-5_ptime_5_seqerr_0.1-0.2_reads_200/split_mapped_reads/timepoints_splitted/readnames/')

library(tidyverse)

pdf_file <- "classification_plots.pdf"
pdf(pdf_file)

for (XX in 5:6) {
  for (YY in 5:5) {
    for (ZZ in 5:5) {
      target_dir <- paste0("5eUrates_", XX, "-", XX, "_6sGrates_", YY, "-", YY, "_4sUrates_", ZZ, "-", ZZ, "_ptime_5_seqerr_0.1-0.2_reads_200")
      source_dir <- file.path(parent_dir, target_dir, "split_mapped_reads", "timepoints_splitted", "readnames")
      setwd(source_dir)
      all_files = list.files(pattern=".readnames.txt")
      prop_table <- data.frame()

      for (file in all_files){
        test <- read.table(file=file,sep = '\t',header=F,stringsAsFactors = FALSE)
        gene <- gsub(".*[_]([^.]+)[_].*", "\\1", file) #Get gene name from file name
  
        classification_timepoint <- gsub(".*[_]([^.]+)[.].*", "\\1", file) #Get classification timepoint from the file name
        real_timepoint <- gsub(".*_", "\\1", test$V1) #Get real timepoint from column values
  
        temp_prop_table <- prop.table(table(real_timepoint)) %>% as.data.frame() #Get frequencies for each timepoint
        temp_prop_table$classification_timepoint <- classification_timepoint
  
        temp_prop_table$gene <- gene

        prop_table <- rbind(prop_table,temp_prop_table)
  
      }


      prop_table$real_timepoint <- factor(prop_table$real_timepoint,levels = c('0-5min','5-10min','10-15min')) #Factor to make the plot more readable
      prop_table$classification_timepoint <- factor(prop_table$classification_timepoint,levels = c('0-5','5-10','10-15'),labels = c('0-5min','5-10min','10-15min'))

      ## Add in details of gene
      #refbed <- read.table(ref, sep="", fill=TRUE, header=FALSE)
      refbed <- read.table('/pi/athma.pai-umw/genomes/hg38/Homo_sapiens.GRCh38.95.uniquegene.bed.bed',sep="",fill=TRUE, header=F) 
      colnames(refbed) <- c('chrom', 'chromStart', 'chromEnd', 'gene', 'na', 'strand')
      refgenome <- subset(refbed, select = -c(na))
      refgenome$geneLength <- refgenome$chromEnd - refgenome$chromStart
      refgenome <- refgenome[, c(1,2,3,6,4,5)]

      prop_table <- merge(prop_table,refgenome[,c('gene','geneLength', 'strand')],by='gene')

      prop_table$geneLength_deciles <- with(prop_table, cut(geneLength, breaks=quantile(geneLength, probs=seq(0,1, by=0.1), na.rm=TRUE),labels = (1:10), include.lowest=TRUE))

      # Add in details on whether the classification is accurate (eg. frequency of 0-5min reads classified as 0-5min) or inaccurate (eg. frequency of 0-5min reads classified as 5-10min or 10-15min)
      prop_table$classif_correct <- 'Inaccurate'
      prop_table$classif_correct[which(prop_table$real_timepoint==prop_table$classification_timepoint)] <- 'Accurate'

      summary_plot <- ggplot(prop_table) +
        geom_boxplot(aes(x=classif_correct,y=Freq,fill=classification_timepoint,alpha=classif_correct))+
        theme_minimal(base_size = 25) +
        labs(title=paste0("Simulated KB reads with ", XX, "% 5eU, ", YY, "% 6sG, ", ZZ, "% 4sU, ", "200 transcripts per gene"),x='Classification to True Timepoint',fill='Classification timepoint',y='Proportion of reads\nclassified correctly')+
        scale_fill_manual(values = c("#00A2FF",'#FF5530','#6CEB76'))+
        facet_wrap(.~classification_timepoint)+
        scale_alpha_manual(values=c(1,0.5))+
        theme(legend.position = 'none')+
        scale_y_continuous(breaks = c(seq(0,1,0.2)))
      summary_plot
    }
  }
}

#setwd(final_dir)
#dev.off()

# move to within for-loop above
#if (XX < 15 | YY < 15 | ZZ < 15) {
#dev.new()
#print('running')
#}
