library(stats)

# read in attributes
args = commandArgs(trailingOnly = T)
DIR=args[1] #directory
U_dist = as.numeric(args[2]) # gene length 1-50kb
RTR = as.numeric(args[3]) # RTR distance 1-100kb
expression_level = as.numeric(args[4]) # expression level

# cleavage rate model
sumsqequationsolve <- function(atts, txnrate){
  # atts[1:5] are Dprimes
  # atts[6:10] are Rprimes
  D_prime = atts[1:3]
  R_prime = atts[4:6]
 # D_prime = atts[1]
 # R_prime = atts[2]
  hold.row <- c(NA, NA)
  f <- function(h){ ((h*(1-2^(-D_prime[1]/(h*txnrate)))) - R_prime[1])^2  + 
                    ((h*(1-2^(-D_prime[2]/(h*txnrate)))) - R_prime[2])^2 + 
                    ((h*(1-2^(-D_prime[3]/(h*txnrate)))) - R_prime[3])^2} 
                    
                 #   +
                  #  ((h*(1-2^(-D_prime[4]/(h*txnrate)))) - R_prime[4])^2 +
                   # ((h*(1-2^(-D_prime[5]/(h*txnrate)))) - R_prime[5])^2 }
  starth = 0
  if(sum(is.na(R_prime))==3){ return(hold.row) } # ==3 & ==1 when only one label
  try(fit.hold <- optim(starth, f))
  try(hold.row <- c(fit.hold$par, fit.hold$value))
  return(hold.row)
}

# Fragment and Read Function
get_reads <- function(lengths, eta_val, insertsize)
{
  # Select a fragment from each transcript, size select, and return the starting position of the
  # resulting reads relative to the length of the transcript
  # Inputs:
  #   lengths - the lengths of the transcript
  #   eta_val - the eta value input to the Weibull distribution - lower bound insertsize
  #   insertsize - a list of length two with the lower and upper bound for fragments - mean +- sd
  # Outputs:
  #   fragments - a data frame with the start positions of the filtered fragments and the index
  #               of the transcript that it came from (columns: transcript, start)

  # sample lengths from a weibull distribution for each transcript and transform them to the length
  # of the transcripts
  deltas = log10(lengths)
  ns_minus_1 <- pmax(round(lengths/eta_val/gamma(1/deltas + 1)) - 1, 0)
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

  # convert to a data frame of fragments and associated transcritp index
  #fragments = data.frame(transcript = rep(1:length(deltas), lengths(delta_is)),
  fragments = data.frame(transcript = rep(1:length(deltas), unlist(lapply(delta_is, length))),
                         start = unlist(starts),
                         end = unlist(ends))
  fragments$length = fragments$end - fragments$start

  # Filter fragments by length and return
  fragments = fragments[fragments$length >= insertsize[1] & fragments$length <= insertsize[2],]
  return(fragments)
}

simulate <- function(U_dist, RTR, labeling, half_life,
                     expression_level, n_millions, transcription_rate,
                     mean_insert, sd_insert)
  {
  # Inputs:
  #   U_dist - distance from the TSS to cleavage site, in kb
  #   RTR - read through region, distance from the cleavage site to the transcription end site, in kb
  #   labeling - the length of the labeling period in minutes
  #   half_life - the half-life time in minutes
  #   expression_level - the expression level in TPM
  #   n_millions - the total number of millions of transcripts to consider
  #   transcription_rate - rate of transcription in kb/min
  #   mean_insert - mean fragment size of fragment size selection, in nt
  #   sd_insert - std. dev. of the fragment size distribution, in nt
  # Outputs: (in vector form)
  #   [1] true # of total transcripts
  #   [2] true # beforeCS transcripts
  #   [3] true # of uncleaved transcripts
  #   [4] true # of cleaved transcripts
  #   [5] count of total reads
  #   [6] count of uncleaved reads

  U_dist = 1000*U_dist
  RTR = 1000*RTR
  transcription_rate = 1000*transcription_rate

  ### Generate expression_level*n_millions transcripts uniformly from the labeled region
  # number molecules determined by density of PolII on DNA: # polII / kb of gene+RTR
  txndistance = U_dist + RTR
  end_sites = sample.int(txndistance + labeling*transcription_rate,
                                  expression_level*n_millions*(txndistance/1000), replace = TRUE)

  ### Determine if the transcripts are cleaved or not and the resulting lengths of the molecules generated
  # get molecules past CS (> U_dist)
  pastCS = which(end_sites > U_dist)
  # calculate cleaved molecules, subtract U_dist from end_sites thus setting CS as 0 (as parametrized)   
  cleaved_pastCS = (runif(length(end_sites[pastCS]))>(2^(-(end_sites[pastCS]-U_dist)/(half_life*transcription_rate))))
  # create empty vector for all end_sites and then fill in status of those past CS
  cleaved = rep(FALSE, length(end_sites))
  cleaved[pastCS] <- cleaved_pastCS

  # divide into uncleaved and cleaved fractions
  uncleaved_lengths = pmin(end_sites[-which(cleaved)], U_dist+RTR)
  cleaved_mature_lengths = rep(U_dist, sum(cleaved))
  cleaved_RTR_lengths = pmin(end_sites[which(cleaved)], U_dist+RTR)-U_dist

  ### get true numbers
  true_total = length(which(uncleaved_lengths > U_dist - (mean_insert + sd_insert) - rl)) +
               length(cleaved_mature_lengths) # ALL transcripts past U_dist - (mean_insert - sd_insert)
  true_beforeCS = length(which(uncleaved_lengths > U_dist - (mean_insert + sd_insert) - rl & 
  	                           uncleaved_lengths < U_dist)) # ALL transcripts between U_dist - (mean_insert - sd_insert) & U_dist
  true_uncleaved =  length(which(uncleaved_lengths > U_dist)) # uncleaved transcripts past U_dist
  true_cleaved = length(cleaved_mature_lengths) # transcripts past U_dist and cleaved

  ### Get the reads from the transcripts and map them to the gene
  rl = 50

  # get reads uncleaved molecules (start at 0)
  uncleaved_start_pos <- get_reads(uncleaved_lengths, (mean_insert-sd_insert), c((mean_insert-sd_insert), (mean_insert+sd_insert))) #assumption is that all transcript start at same position (0)
  if (nrow(uncleaved_start_pos) >0){ uncleaved_start_pos$cleaved = "uncleaved" }
  uncleaved_start_pos$start_coord = uncleaved_start_pos$start
  uncleaved_start_pos$end_coord = uncleaved_start_pos$end
    
  # get reads from cleaved mature (start at 0)
  cleaved_mature_start_pos <- get_reads(cleaved_mature_lengths, (mean_insert-sd_insert), c((mean_insert-sd_insert), (mean_insert+sd_insert))) #assumption is that all transcript start at same position (0)
  if (nrow(cleaved_mature_start_pos) >0){ cleaved_mature_start_pos$cleaved = "cleaved_mature"}
  cleaved_mature_start_pos$start_coord = cleaved_mature_start_pos$start    
  cleaved_mature_start_pos$end_coord = cleaved_mature_start_pos$end    
    
  # get reads from cleaved read through (start at U_dist)
  cleaved_RTR_start_pos <- get_reads(cleaved_RTR_lengths, (mean_insert-sd_insert), c((mean_insert-sd_insert), (mean_insert+sd_insert))) #assumption is that all transcript start at same position (0)
  if (nrow(cleaved_RTR_start_pos) >0){ cleaved_RTR_start_pos$cleaved = "cleaved_RTR"}
  cleaved_RTR_start_pos$start_coord = cleaved_RTR_start_pos$start + U_dist    
  cleaved_RTR_start_pos$end_coord = cleaved_RTR_start_pos$end + U_dist

  # combine all fragments
  reads = rbind(uncleaved_start_pos, cleaved_mature_start_pos, cleaved_RTR_start_pos)

  # subsample reads to # fragments calculated from FPKM
  #reads_all = rbind(uncleaved_start_pos, cleaved_mature_start_pos, cleaved_RTR_start_pos)
  #n_fragments = (expression_level * (U_dist/1000))*n_millions
  #reads = reads_all[sample.int(nrow(reads_all), n_fragments, replace=T),]

  ### Count reads in various regions
  # reads over cleavage site (uncleaved)
  uncleaved_num = nrow(subset(reads, start_coord > U_dist - rl + 10 & start_coord <= U_dist -10)) +
  				  nrow(subset(reads, end_coord > U_dist + 10 & end_coord <= U_dist + rl - 10))

  # reads in total region
  total_num = nrow(subset(reads, start_coord > U_dist - mean_insert - (2*sd_insert) - rl + 10 & 
  	                             start_coord <= U_dist - mean_insert - (2*sd_insert) - 10)) +
              nrow(subset(reads, end_coord > U_dist - mean_insert - (2*sd_insert) + 10 & 
  	                             end_coord <= U_dist - mean_insert - (2*sd_insert) + rl - 10))

  # return items
     return(c(true_total, true_beforeCS, true_uncleaved, true_cleaved, total_num, uncleaved_num))
}

##### Grid of paramters to simulate over #####
## keep constant
n_millions = 100
rl = 50

## variables to simulate over
labelings = c(5, 10, 20)
#half_lives = c(seq(0.05,0.9,by=0.05),seq(1,10,by=0.75))
half_lives = c(seq(0.05, 0.9, by=0.1), seq(1, 10, 1))
mean_inserts = c(125, 150, 175, 200)
#mean_inserts = 250
#sd_inserts = c(25, 50)
sd_inserts = 30
#transcription_rates = seq(0.5, 4, 0.5)
transcription_rates = 1.5

##### Run the grid simulation #####
print("RUNNING SIMULATIONS")

# initiate dataframe
simulation_results = c()
# loop through parameters
for (meand in mean_inserts){
	for (sdd in sd_inserts){
		for (txnrate in transcription_rates){
			for (hl in half_lives){
				for (L in labelings){
					 print(paste0("minute ",L," - distance ",RTR," - expression ",expression_level," - half-life ",hl,"..."))
                     results = simulate(U_dist, RTR, L, hl, expression_level, n_millions, txnrate, meand, sdd)
                     # get read count values
                     total_num = results[5]
                     uncleaved_num = results[6]
                     #beforeCS_num = ((meand + sdd)/(meand + sdd + (RTR*1000) + L + (txnrate * 1000)))*total_num
                     ### CHANGE - remove L = (L*(txnrate*1000))
                     beforeCS_num = ((meand + sdd)/(meand + sdd + (RTR*1000) + L + (txnrate * 1000)))*(total_num-uncleaved_num)
                     cleaved_num = total_num - uncleaved_num - beforeCS_num
                     # bind to dataframe
                     simulation_results = rbind(simulation_results,
                     	                        data.frame(U_dist = U_dist, RTR = RTR, TPM = expression_level,
                     	                                   n_millions = n_millions, readlength = rl,
                     	                                   mean_insert = meand, sd_insert = sdd,
                     	                                   transcription_rate = txnrate, labeling = L,
                     	                                   half_life = hl,
                     	                                   # true
                     	                                   true_total = results[1],
                     	                                   true_beforeCS = results[2],
                     	                                   true_uncleaved = results[3],
                     	                                   true_cleaved = results[4],
                     	                                   # counts
                     	                                   total_num = total_num,
                     	                                   beforeCS_num = beforeCS_num,
                     	                                   uncleaved_num = uncleaved_num,
                     	                                   cleaved_num = cleaved_num))
				}
			}
		}
	}
}

# write counts table
print("Writing simulation read table")
simfile = paste0(DIR, "/countsSIM_U",U_dist,"_RTR",RTR,"_X",expression_level,".txt")
mkdir(DIR)
write.table(simulation_results, file=simfile, sep="\t", quote=F, row.names=F, col.names=T)

### Calculate and reformat data

print("Reformatting simulation data")
### add parameters needed for running model on true data
simulation_results$true_ratio = simulation_results$true_uncleaved/simulation_results$true_cleaved
simulation_results$true_D_prime <- (simulation_results$RTR) + (simulation_results$labeling*simulation_results$transcription_rate) # RTR+Lr
simulation_results$true_R_prime <- (simulation_results$true_D_prime*log(2)) / (simulation_results$transcription_rate*((1/simulation_results$true_ratio) + 1))

# add parameters needed for running model on count data
simulation_results$ratio = simulation_results$uncleaved_num/simulation_results$cleaved_num
simulation_results$D_prime <- (simulation_results$RTR) + (simulation_results$labeling*simulation_results$transcription_rate) # RTR+Lr
simulation_results$R_prime <- (simulation_results$D_prime*log(2)) / (simulation_results$transcription_rate*((1/simulation_results$ratio) + 1))

# reformat to make wide table
simulation_results_5 <- subset(simulation_results, labeling==5)
simulation_results_10 <- subset(simulation_results, labeling==10)
simulation_results_20 <- subset(simulation_results, labeling==20)

results_wide <- simulation_results_5[,c(1:10)]

results_wide <- data.frame(results_wide,
	true_uncleaved_5 = simulation_results_5$true_uncleaved, true_uncleaved_10 = simulation_results_10$true_uncleaved, true_uncleaved_20 = simulation_results_20$true_uncleaved, 
	true_cleaved_5 = simulation_results_5$true_cleaved, true_cleaved_10 = simulation_results_10$true_cleaved, true_cleaved_20 = simulation_results_20$true_cleaved, 
	count_uncleaved_5 = simulation_results_5$uncleaved_num, count_uncleaved_10 = simulation_results_10$uncleaved_num, count_uncleaved_20 = simulation_results_20$uncleaved_num, 
	count_cleaved_5 = simulation_results_5$cleaved_num, count_cleaved_10 = simulation_results_10$cleaved_num, count_cleaved_20 = simulation_results_20$cleaved_num, 
	true_ratio_5 = simulation_results_5$true_ratio, true_ratio_10 = simulation_results_10$true_ratio, true_ratio_20 = simulation_results_20$true_ratio,
	count_ratio_5 = simulation_results_5$ratio, count_ratio_10 = simulation_results_10$ratio, count_ratio_20 = simulation_results_20$ratio,
	true_Dprime_5 = simulation_results_5$true_D_prime, true_Dprime_10 = simulation_results_10$true_D_prime, true_Dprime_20 = simulation_results_20$true_D_prime,
	count_Dprime_5 = simulation_results_5$D_prime, count_Dprime_10 = simulation_results_10$D_prime, count_Dprime_20 = simulation_results_20$D_prime,
	true_Rprime_5 = simulation_results_5$true_R_prime, true_Rprime_10 = simulation_results_10$true_R_prime, true_Rprime_20 = simulation_results_20$true_R_prime,
	count_Rprime_5 = simulation_results_5$R_prime, count_Rprime_10 = simulation_results_10$R_prime, count_Rprime_20 = simulation_results_20$R_prime)


### Running Cleavage Rate model

print("Calculting true cleavage rates")
true_sumsqfit.data <- t(apply(results_wide[,c('true_Dprime_5', 'true_Dprime_10', 'true_Dprime_20', 'true_Rprime_5', 'true_Rprime_10', 'true_Rprime_20')], 
	1, sumsqequationsolve, results_wide$transcription_rate[1]))
results_wide$true_hl = true_sumsqfit.data[,1]

print("Calculating simulated cleavage rates")
count_sumsqfit.data <- t(apply(results_wide[,c('count_Dprime_5', 'count_Dprime_10', 'count_Dprime_20', 'count_Rprime_5', 'count_Rprime_10', 'count_Rprime_20')], 
	1, sumsqequationsolve, results_wide$transcription_rate[1]))
results_wide$count_hl = count_sumsqfit.data[,1]

print("Writing out half lives")
juncfile = paste0(DIR,"/juncSIM_U",U_dist,"_RTR",RTR,"_X",expression_level,".txt")
mkdir(DIR)
write.table(results_wide, file=juncfile, sep="\t", quote=F, row.names=F, col.names=T)

print("All done!")





