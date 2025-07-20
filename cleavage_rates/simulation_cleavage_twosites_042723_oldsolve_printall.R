library(stats)

# read in attributes
args = commandArgs(trailingOnly = T)
DIR=args[1] #directory
U_dist = as.numeric(args[2]) # gene length
RTR = as.numeric(args[3]) # RTR distance
expression_level = as.numeric(args[4]) # expression level
PAU = as.numeric(args[5]) # PAU
extUTR = as.numeric(args[6]) # distance between CS1 and CS2

# cleavage rate model
sumsqequationsolve <- function(atts, txnrate){
  # atts[1:5] are Dprimes
  # atts[6:10] are Rprimes
  D_prime = atts[1:3]
  R_prime = atts[4:6]
 # D_prime = atts[1]
 # R_prime = atts[2]
  hold.row <- c(NA, NA)
  #f <- function(h){ (h - (h - 2^D_prime[1]/(h*txnrate)) * R_prime[1])^2 + 
 # 					        (h - (h - 2^D_prime[2]/(h*txnrate)) * R_prime[2])^2 +
 # 				          (h - (h - 2^D_prime[3]/(h*txnrate)) * R_prime[3])^2 }
  f <- function(h){ ((h*(1 - 2^(-D_prime[1]/(h*txnrate)))) - R_prime[1])^2 + 
  					        ((h*(1 - 2^(-D_prime[2]/(h*txnrate)))) - R_prime[2])^2 + 
  					        ((h*(1 - 2^(-D_prime[3]/(h*txnrate)))) - R_prime[3])^2 }

  starth = 0
  if(sum(is.na(R_prime))==3){ return(hold.row) } # ==3 & ==1 when only one label
  try(fit.hold <- optim(starth, f))
  try(hold.row <- c(fit.hold$par, fit.hold$value))
  return(hold.row)
}

#sum(h - (h - 2^(Dprime/hr) * Rprime))^2, where Rprime = ((ln(2) * ratio)/ (r * u))

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
  if(length(deltas) == 0){
    fragments <- data.frame(matrix(ncol = 4, nrow = 0))
    colnames(fragments) <- c("transcript", "start", "end", "length")
  }

  if(length(deltas) > 0){
    fragments = data.frame(transcript = rep(1:length(deltas), unlist(lapply(delta_is, length))),
                           start = unlist(starts),
                           end = unlist(ends))
    fragments$length = fragments$end - fragments$start
    # Filter fragments by length
    fragments = fragments[fragments$length >= insertsize[1] & fragments$length <= insertsize[2],]
  }

  # return
  return(fragments)
}

### Determine if the transcripts are cleaved or not and the resulting lengths of the molecules generated
cleavageState <- function(end_sites, U_dist, half_life, txnrate){
  # get molecules past CS (>U_dist)
  pastCS = which(end_sites > U_dist)
  # calculate cleaved molecules, subtract U_dist from end_sites, thus setting CS as 0
  cleaved_pastCS = (runif(length(end_sites[pastCS]))>(2^(-(end_sites[pastCS]-U_dist)/(half_life*txnrate))))
  # create empty vector for all end_sites and then fill in status of those past CS
  cleaved = rep(FALSE, length(end_sites))
  cleaved[pastCS] <- cleaved_pastCS
  return(cleaved)
}

### Get the reads from the transcripts and map them to the gene
uncleavedReads <- function(uncleaved_lengths, mean_insert, sd_insert){
  # get reads uncleaved molecules (start at 0)
  uncleaved_start_pos <- get_reads(uncleaved_lengths, (mean_insert-sd_insert), c((mean_insert-sd_insert), (mean_insert+sd_insert))) #assumption is that all transcript start at same position (0)
  if (nrow(uncleaved_start_pos) >0){ uncleaved_start_pos$cleaved = "uncleaved" }
  uncleaved_start_pos$start_coord = uncleaved_start_pos$start
  uncleaved_start_pos$end_coord = uncleaved_start_pos$end
  return(uncleaved_start_pos)
}

cleavedmatureReads <- function(cleaved_mature_lengths, mean_insert, sd_insert){
  # get reads from cleaved mature (start at 0)
  cleaved_mature_start_pos <- get_reads(cleaved_mature_lengths, (mean_insert-sd_insert), c((mean_insert-sd_insert), (mean_insert+sd_insert))) #assumption is that all transcript start at same position (0)
  if (nrow(cleaved_mature_start_pos) >0){ cleaved_mature_start_pos$cleaved = "cleaved_mature"}
  cleaved_mature_start_pos$start_coord = cleaved_mature_start_pos$start   
  cleaved_mature_start_pos$end_coord = cleaved_mature_start_pos$end    
  return(cleaved_mature_start_pos)
}

cleavedRTRReads <- function(cleaved_RTR_lengths, mean_insert, sd_insert, U_dist){
  # get reads from cleaved read through (start at U_dist)
  cleaved_RTR_start_pos <- get_reads(cleaved_RTR_lengths, (mean_insert-sd_insert), c((mean_insert-sd_insert), (mean_insert+sd_insert))) #assumption is that all transcript start at same position (0)
  if (nrow(cleaved_RTR_start_pos) >0){ cleaved_RTR_start_pos$cleaved = "cleaved_RTR"}
  cleaved_RTR_start_pos$start_coord = cleaved_RTR_start_pos$start + U_dist
  cleaved_RTR_start_pos$end_coord = cleaved_RTR_start_pos$end + U_dist
  return(cleaved_RTR_start_pos)
}

countReads <- function(reads, U_dist, rl, mean_insert, sd_insert){
  ### Count reads in various regions
  # reads over cleavage site (uncleaved)
  uncleaved_num = nrow(subset(reads, start_coord > U_dist - rl + 10 & start_coord <= U_dist -10)) +
    nrow(subset(reads, end_coord > U_dist + 10 & end_coord <= U_dist + rl - 10))
  # reads in total region
  total_num = nrow(subset(reads, start_coord > U_dist - mean_insert - (2*sd_insert) - rl + 10 &
                            start_coord <= U_dist - mean_insert - (2*sd_insert) - 10)) +
    nrow(subset(reads, end_coord > U_dist - mean_insert - (2*sd_insert) + 10 &
                  end_coord <= U_dist - mean_insert - (2*sd_insert) + rl - 10))
  return(c(total_num, uncleaved_num))
}

trueNumbers <- function(uncleaved_lengths, cleaved_mature_lengths, U_dist, mean_insert, sd_insert, rl){
  ### get true numbers
  true_total = length(which(uncleaved_lengths > U_dist - (mean_insert + sd_insert) - rl)) +
               length(cleaved_mature_lengths) # ALL transcripts past U_dist - (mean_insert - sd_insert)
  true_beforeCS = length(which(uncleaved_lengths > U_dist - (mean_insert + sd_insert) - rl & 
                               uncleaved_lengths < U_dist)) # ALL transcripts between U_dist - (mean_insert - sd_insert) & U_dist
  true_uncleaved =  length(which(uncleaved_lengths > U_dist)) # uncleaved transcripts past U_dist
  true_cleaved = length(cleaved_mature_lengths) # transcripts past U_dist and cleaved
  return(c(true_total, true_beforeCS, true_uncleaved, true_cleaved))
}

## add PAU, extUTR, hl1, hl2
simulate <- function(U_dist, RTR, labeling, 
                     PAU, extUTR, hl1, hl2,
                     expression_level, n_millions, transcription_rate,
                     mean_insert, sd_insert)
  {
  # Inputs:
  #   U_dist - distance from the TSS to cleavage site, in kb
  #   RTR - read through region, distance from the cleavage site to the transcription end site, in kb
  #   labeling - the length of the labeling period in minutes
  #   PAU - proportion of transcripts that use CS1
  #   extUTR - distance between CS1 and CS2
  #   hl1 - half-life of CS1 in minutes
  #   hl2 - half-life of CS2 in minutes
  #   expression_level - the expression level in TPM
  #   n_millions - the total number of millions of transcripts to consider
  #   transcription_rate - rate of transcription in kb/min
  #   mean_insert - mean fragment size of fragment size selection, in nt
  #   sd_insert - std. dev. of the fragment size distribution, in nt
  # Outputs: (in vector form)
  #   [1] true # of total transcripts for CS1
  #   [2] true # beforeCS transcripts for CS1
  #   [3] true # of uncleaved transcripts for CS1
  #   [4] true # of cleaved transcripts for CS1
  #   [5] count of total reads for CS1
  #   [6] count of uncleaved reads for CS2
  #   [7] true # of total transcripts for CS1
  #   [8] true # beforeCS transcripts for CS1
  #   [9] true # of uncleaved transcripts for CS1
  #   [10] true # of cleaved transcripts for CS1
  #   [11] count of total reads for CS1
  #   [12] count of uncleaved reads for CS2

  U_dist = 1000*U_dist
  RTR = 1000*RTR
  transcription_rate = 1000*transcription_rate

  ### Generate expression_level*n_millions transcripts uniformly from the labeled region
  # number molecules determined by density of PolII on DNA: # polII / kb of gene+RTR
  txndistance = U_dist + extUTR + RTR
  end_sites = sample.int(txndistance + labeling*transcription_rate,
                                  expression_level*n_millions*(txndistance/1000), replace = TRUE)

  ### Divide transcripts in CS1 vs CS2 proportionally based on PAU
  inds <- seq(1:length(end_sites))
  cs1inds <- sample(inds, length(end_sites)*PAU, replace=F)
  if(length(cs1inds) == 0){ cs2inds <- inds }
  if(length(cs1inds) > 0){ cs2inds <- inds[-cs1inds] }

  cs1_end_sites <- end_sites[cs1inds]
  cs2_end_sites <- end_sites[cs2inds]

  ### Determine if the transcripts are cleaved or not and the resulting lengths of the molecules generated
  # get cleaved sites for each CS
  cs1_cleaved <- cleavageState(cs1_end_sites, U_dist, hl1, transcription_rate)
  cs2_cleaved <- cleavageState(cs2_end_sites, U_dist + extUTR, hl2, transcription_rate)

  # divide into uncleaved and cleaved fractions
  cs1_uncleaved_lengths = pmin(cs1_end_sites[-which(cs1_cleaved)], U_dist+extUTR+RTR)
  cs2_uncleaved_lengths = pmin(cs2_end_sites[-which(cs2_cleaved)], U_dist+extUTR+RTR)

  cs1_cleaved_mature_lengths = rep(U_dist, sum(cs1_cleaved))
  cs2_cleaved_mature_lengths = rep(U_dist+extUTR, sum(cs2_cleaved))

  cs1_cleaved_RTR_lengths = pmin(cs1_end_sites[which(cs1_cleaved)], U_dist+extUTR+RTR)-U_dist
  cs2_cleaved_RTR_lengths = pmin(cs2_end_sites[which(cs2_cleaved)], U_dist+extUTR+RTR)-(U_dist+extUTR)

  ### Get the reads from the transcripts and map them to the gene
  rl = 50

  # cs1 reads
  cs1_uncleaved_start_pos = uncleavedReads(cs1_uncleaved_lengths, mean_insert, sd_insert)
  cs1_cleaved_mature_start_pos = cleavedmatureReads(cs1_cleaved_mature_lengths, mean_insert, sd_insert)
  cs1_cleaved_RTR_start_pos = cleavedRTRReads(cs1_cleaved_RTR_lengths, mean_insert, sd_insert, U_dist)

  cs1_reads = rbind(cs1_uncleaved_start_pos, cs1_cleaved_mature_start_pos, cs1_cleaved_RTR_start_pos)
  cs1_reads = subset(cs1_reads[-which(cs1_reads$cleaved == "cleaved_RTR"),])

  # cs1 true numbers
  cs1_trueCount = trueNumbers(cs1_uncleaved_lengths, cs1_cleaved_mature_lengths, U_dist, mean_insert, sd_insert, rl)

  # cs2 reads
  cs2_uncleaved_start_pos = uncleavedReads(cs2_uncleaved_lengths, mean_insert, sd_insert)
  cs2_cleaved_mature_start_pos = cleavedmatureReads(cs2_cleaved_mature_lengths, mean_insert, sd_insert)
  cs2_cleaved_RTR_start_pos = cleavedRTRReads(cs2_cleaved_RTR_lengths, mean_insert, sd_insert, U_dist+extUTR)

  cs2_reads = rbind(cs2_uncleaved_start_pos, cs2_cleaved_mature_start_pos, cs2_cleaved_RTR_start_pos)
  cs2_reads = subset(cs2_reads[-which(cs2_reads$cleaved == "cleaved_RTR"),])
  # cs2 true numbers
  cs2_trueCount = trueNumbers(cs2_uncleaved_lengths, cs2_cleaved_mature_lengths, U_dist+extUTR, mean_insert, sd_insert, rl)

  # count only cs1 and only cs2
  cs1ONLY_readCount_mature = countReads(cs1_cleaved_mature_start_pos, U_dist, rl, 2*mean_insert, sd_insert)
  cs1ONLY_readCount_uncleaved = countReads(cs1_uncleaved_start_pos, U_dist, rl, 2*mean_insert, sd_insert)
  cs2ONLY_readCount_mature = countReads(cs2_cleaved_mature_start_pos, U_dist+extUTR, rl, 2*mean_insert, sd_insert)
  cs2ONLY_readCount_uncleaved = countReads(cs2_uncleaved_start_pos, U_dist+extUTR, rl, 2*mean_insert, sd_insert)

  # have to be agnostic to cs1 and cs2 when counting reads!
  all_reads = rbind(cs1_reads, cs2_reads)
  #all_reads = subset(all_reads[-which(all_reads$cleaved == "cleaved_RTR"),])
  cs1_readCount = countReads(all_reads, U_dist, rl, 2*mean_insert, sd_insert) 
  cs2_readCount = countReads(all_reads, U_dist+extUTR, rl, 2*mean_insert, sd_insert)

  # return items
  # 1 = cs1_true_total, 2 = cs1_true_beforeCS, 3 = cs1_true_uncleaved, 4 = cs1_true_cleaved, 
  # 5 = cs1_total_num, 6 = cs1_uncleaved_num
  # 7 = cs2_true_total, 8 = cs2_true_beforeCS, 9 = cs2_true_uncleaved, 10 = cs2_true_cleaved, 
  # 11 = cs2_total_num, 12 = cs1_uncleaved_num
  # 13 = cs1ONLYmature_total, 14 = cs1ONLYmature_uncleaved,
  # 15 = cs1ONLYuncleaved_total, 16 = cs2ONLYuncleaved_uncleaved,
  # 17 = cs2ONLYmature_total, 18 = cs2ONLYmature_uncleaved,
  # 19 = cs2ONLYuncleaved_total, 20 = cs2ONLYuncleaved_uncleaved

  return(c(cs1_trueCount, cs1_readCount, cs2_trueCount, cs2_readCount, 
           cs1ONLY_readCount_mature, cs1ONLY_readCount_uncleaved,
           cs2ONLY_readCount_mature, cs2ONLY_readCount_uncleaved))
}

##### Grid of paramters to simulate over #####
## keep constant
n_millions = 100
rl = 50

## variables to simulate over
labelings = c(5, 10, 20)
half_lives = c(seq(0.1,0.9,by=0.1),seq(1,10,by=2))
mean_inserts = 165
sd_inserts = 30
#transcription_rates = seq(0.5, 4, 0.5)
transcription_rates = 1.5

# make into input arguments
#PAUS = seq(0.1, 0.9, 0.05)
#extensions = c(seq(100, 1000, by=100), seq(1250, 2000, by=250))

##### Run the grid simulation #####
print("RUNNING SIMULATIONS")

# make variables constant and remove from loop
meand = mean_inserts
sdd = sd_inserts
txnrate = transcription_rates

# initiate dataframe
simulation_results = c()
# loop through parameters
for (hl1 in half_lives){
	for (hl2 in half_lives){
		for (L in labelings){
			print(paste0("hl1 ",hl1," - hl2 ",hl2, " - L ", L, "..."))
			results = simulate(U_dist, RTR, L, PAU, extUTR, hl1, hl2, expression_level, n_millions, txnrate, meand, sdd)
			# get read count values
        	cs1_total_num = results[5]
	        cs1_uncleaved_num = results[6]
	        cs2_total_num = results[11]
    	    cs2_uncleaved_num = results[12]
          ### CS1 calculations ###
          # scale TOTAL reads by cs1 PAU (before all CS)
          cs1_total = cs1_total_num * PAU
          # calculate cs1 BEFORE cs (fractional distance of mean + sd)
          cs1_before_fraction = ((2*meand + (2*sdd))/(2*meand + (2*sdd) + (extUTR + (RTR*1000)) + (L * (txnrate*1000))))
          # calculate cs1 UNCLEAVED
          cs1_uncleaved = max(cs1_uncleaved_num - ((cs1_total_num * (1-PAU)) - (cs1_total_num*cs1_before_fraction)), 0)
          # calculate cs1 BEFORE cs (fractional distance of mean + sd)
          cs1_before = cs1_before_fraction * (cs1_total - cs1_uncleaved)
          # calculate cs1 CLEAVED
          cs1_cleaved = max(cs1_total - cs1_uncleaved - cs1_before, 0)
          ### CS2 calculations ###
          cs2_PAU = 1 - PAU
          # cs2-cs1 intersite distance FRACTION
          interfraction = extUTR/(extUTR + (RTR*1000) + (L * (txnrate * 1000)))
          # scale TOTAL reads by cs2 PAU
          cs2_total = max((cs1_total_num * cs2_PAU) - ((cs1_total_num * interfraction)*cs2_PAU), 0)
          # calculate cs2 UNCLEAVED
          cs2_uncleaved = max(cs2_uncleaved_num - (cs1_uncleaved - (cs1_uncleaved * interfraction)), 0)
          #cs2_uncleaved = cs2_uncleaved_num
          # calculate cs2 BEFORE cs
          cs2_before = max(cs1_before + (interfraction * (cs2_total - cs2_uncleaved)), 0)
          # calculate cs2 CLEAVED
          cs2_cleaved = max(cs2_total - cs2_uncleaved - cs2_before, 0)

	        # calculate cleaved read count
    	    #cs1_beforeCS_num = ((meand + sdd)/(meand + sdd + (extUTR + (RTR*1000)) + (L * (txnrate*1000)))) * (cs1_total_num - cs1_uncleaved_num)
        	#cs1_cleaved_num = cs1_total_num - cs1_uncleaved_num - cs1_beforeCS_num
	        #cs2_beforeCS_num = ((meand + sdd)/(meand + sdd + (RTR*1000) + (L * (txnrate*1000)))) * (cs2_total_num - cs2_uncleaved_num)
    	    #cs2_cleaved_num = cs2_total_num - cs2_uncleaved_num - cs2_beforeCS_num
        	# bind to dataframe
	        simulation_results = rbind(simulation_results,
    	    	data.frame(U_dist = U_dist, RTR = RTR, #1,2
        	               TPM = expression_level, n_millions = n_millions, #3,4
                       	   readlength = rl, mean_insert = meand, sd_insert = sdd, #5,6,7
                           transcription_rate = txnrate, labeling = L, #8,9
                           PAU = PAU, extUTR = extUTR, #10,11
                           cs1_half_life = hl1, cs2_half_life = hl2, #12,13
                           # true
                           cs1_true_total = results[1], #14
                           cs1_true_beforeCS = results[2], #15
                           cs1_true_uncleaved = results[3], #16
                           cs1_true_cleaved = results[4], #17
                           cs2_true_total = results[7], #18
                           cs2_true_beforeCS = results[8], #19
                           cs2_true_uncleaved = results[9], #20
                           cs2_true_cleaved = results[10], #21
                           # site only counts
                           cs1_matureONLY_total = results[13], #22
                           cs1_matureONLY_uncleaved = results[14], #23
                           cs1_uncleavedONLY_total = results[15], #24
                           cs1_uncleavedONLY_uncleaved = results[16], #25
                           cs2_matureONLY_total = results[17], #26
                           cs2_matureONLY_uncleaved = results[18], #27
                           cs2_uncleavedONLY_total = results[19], #28
                           cs2_uncleavedONLY_uncleaved = results[20], #29
                           # raw counts
                           cs1_total_raw = results[5], #30
                           cs1_uncleaved_raw = results[6], #31
                           cs2_total_raw = results[11], #32
                           cs2_uncleaved_raw = results[12], #33
                           # counts
                           cs1_total_num = cs1_total, #34
                           cs1_beforeCS_num = cs1_before, #35
                           cs1_uncleaved_num = cs1_uncleaved, #36
                           cs1_cleaved_num = cs1_cleaved, #37
                           cs2_total_num = cs2_total, #38
                           cs2_beforeCS_num = cs2_before, #39
                           cs2_uncleaved_num = cs2_uncleaved, #40
                           cs2_cleaved_num = cs2_cleaved)) #41
        }
    }
}

# write counts table
print("Writing simulation read table")
simfile = paste0(DIR, "/twocountsSIM_042723_oldsolvePRINT_U",U_dist,"_RTR",RTR,"_X",expression_level,"_PAU",PAU,"_extUTR",extUTR,".txt")
write.table(simulation_results, file=simfile, sep="\t", quote=F, row.names=F, col.names=T)

### Calculate and reformat data

print("Reformatting simulation data")

# separate dataframe into 2 dataframes for CS1 and CS2
# rename columns without "csX_" so that easier to work with for downstream analyses

cs1_simulation_results <- simulation_results[,c(1:11, 12, 14:17, 22:25, 30:31, 34:37)]
cs1_simulation_results$RTR <- cs1_simulation_results$RTR*1000 + cs1_simulation_results$extUTR

cs2_simulation_results <- simulation_results[,c(1:11, 13, 18:21, 26:29, 32:33, 38:41)]
cs2_simulation_results$RTR <- cs2_simulation_results$RTR*1000
cs2_simulation_results$PAU <- 1-cs2_simulation_results$PAU

colnames(cs1_simulation_results) <- colnames(cs2_simulation_results) <- 
   c("U_dist", "RTR", "TPM", "n_millions", "readlength", "mean_insert", "sd_insert",
    "transcription_rate", "labeling", "PAU", "extUTR", "half_life",
    "true_total", "true_beforeCS", "true_uncleaved", "true_cleaved",
    "matureONLY_total", "matureONLY_uncleaved", "uncleavedONLY_total", "uncleavedONLY_uncleaved",
    "total_raw", "uncleaved_raw",
    "count_total", "count_beforeCS", "count_uncleaved", "count_cleaved")

formatWide <- function(sim_results){
  # Dprime is common between two
  sim_results$D_prime <- (sim_results$RTR) + (sim_results$labeling*(sim_results$transcription_rate*1000)) # RTR+Lr
  ### add parameters needed for running model on true data
  sim_results$true_ratio = (sim_results$true_uncleaved/sim_results$true_cleaved)
  sim_results$true_R_prime <- ((log(2) * sim_results$D_prime) / ((sim_results$transcription_rate*1000) * (1/sim_results$true_ratio + 1)))
  ### add parameters needed for running model on raw counts
  sim_results$raw_ratio = (sim_results$uncleavedONLY_uncleaved / sim_results$matureONLY_total)
  sim_results$raw_R_prime <- ((log(2) * sim_results$D_prime) / ((sim_results$transcription_rate*1000) * (1/sim_results$raw_ratio + 1)))
  ### add parameters needed for running model on count data
  sim_results$count_ratio = (sim_results$count_uncleaved/sim_results$count_cleaved)
  sim_results$count_R_prime <- ((log(2) * sim_results$D_prime) / ((sim_results$transcription_rate*1000) * (1/sim_results$count_ratio + 1)))
  # reformat to make wide table
  sim_results_5 <- subset(sim_results, labeling==5)
  sim_results_10 <- subset(sim_results, labeling==10)
  sim_results_20 <- subset(sim_results, labeling==20)
  # make wide dataset
  results_wide <- sim_results_5[,c(1:12)]
  results_wide <- data.frame(results_wide,
    true_uncleaved_5 = sim_results_5$true_uncleaved, true_uncleaved_10 = sim_results_10$true_uncleaved, true_uncleaved_20 = sim_results_20$true_uncleaved, 
    true_total_5 = sim_results_5$true_total, true_total_10 = sim_results_10$true_total, true_total_20 = sim_results_20$true_total, 
    true_beforeCS_5 = sim_results_5$true_beforeCS, true_beforeCS_10 = sim_results_10$true_beforeCS, true_beforeCS_20 = sim_results_20$true_beforeCS, 
    true_cleaved_5 = sim_results_5$true_cleaved, true_cleaved_10 = sim_results_10$true_cleaved, true_cleaved_20 = sim_results_20$true_cleaved, 
    matureONLY_total_5 = sim_results_5$matureONLY_total, matureONLY_total_10 = sim_results_10$matureONLY_total, matureONLY_total_20 = sim_results_20$matureONLY_total, 
    matureONLY_uncleaved_5 = sim_results_5$matureONLY_uncleaved, matureONLY_uncleaved_10 = sim_results_10$matureONLY_uncleaved, matureONLY_uncleaved_20 = sim_results_20$matureONLY_uncleaved, 
    uncleavedONLY_total_5 = sim_results_5$uncleavedONLY_total, uncleavedONLY_total_10 = sim_results_10$uncleavedONLY_total, uncleavedONLY_total_20 = sim_results_20$uncleavedONLY_total, 
    uncleavedONLY_uncleaved_5 = sim_results_5$uncleavedONLY_uncleaved, uncleavedONLY_uncleaved_10 = sim_results_10$uncleavedONLY_uncleaved, uncleavedONLY_uncleaved_20 = sim_results_20$uncleavedONLY_uncleaved, 
    raw_total_5 = sim_results_5$total_raw, raw_total_10 = sim_results_10$total_raw, raw_total_20 = sim_results_20$total_raw, 
    raw_uncleaved_5 = sim_results_5$uncleaved_raw, raw_uncleaved_10 = sim_results_10$uncleaved_raw, raw_uncleaved_20 = sim_results_20$uncleaved_raw, 
    count_uncleaved_5 = sim_results_5$count_uncleaved, count_uncleaved_10 = sim_results_10$count_uncleaved, count_uncleaved_20 = sim_results_20$count_uncleaved, 
    count_total_5 = sim_results_5$count_total, count_total_10 = sim_results_10$count_total, count_total_20 = sim_results_20$count_total, 
    count_beforeCS_5 = sim_results_5$count_beforeCS, count_beforeCS_10 = sim_results_10$count_beforeCS, count_beforeCS_20 = sim_results_20$count_beforeCS, 
    count_cleaved_5 = sim_results_5$count_cleaved, count_cleaved_10 = sim_results_10$count_cleaved, count_cleaved_20 = sim_results_20$count_cleaved, 
    true_ratio_5 = sim_results_5$true_ratio, true_ratio_10 = sim_results_10$true_ratio, true_ratio_20 = sim_results_20$true_ratio,
    raw_ratio_5 = sim_results_5$raw_ratio, raw_ratio_10 = sim_results_10$raw_ratio, raw_ratio_10 = sim_results_10$raw_ratio, 
    count_ratio_5 = sim_results_5$count_ratio, count_ratio_10 = sim_results_10$count_ratio, count_ratio_20 = sim_results_20$count_ratio,
    true_Dprime_5 = sim_results_5$D_prime, true_Dprime_10 = sim_results_10$D_prime, true_Dprime_20 = sim_results_20$D_prime,
    raw_Dprime_5 = sim_results_5$D_prime, raw_Dprime_10 = sim_results_10$D_prime, raw_Dprime_20 = sim_results_20$D_prime,
    count_Dprime_5 = sim_results_5$D_prime, count_Dprime_10 = sim_results_10$D_prime, count_Dprime_20 = sim_results_20$D_prime,
    true_Rprime_5 = sim_results_5$true_R_prime, true_Rprime_10 = sim_results_10$true_R_prime, true_Rprime_20 = sim_results_20$true_R_prime,
    raw_Rprime_5 = sim_results_5$raw_R_prime, raw_Rprime_10 = sim_results_10$raw_R_prime, raw_Rprime_20 = sim_results_20$raw_R_prime, 
    count_Rprime_5 = sim_results_5$count_R_prime, count_Rprime_10 = sim_results_10$count_R_prime, count_Rprime_20 = sim_results_20$count_R_prime)
  # return
  return(results_wide)
}

cs1_results_wide <- formatWide(cs1_simulation_results)
cs2_results_wide <- formatWide(cs2_simulation_results)


### Running Cleavage Rate model

print("Calculting true cleavage rates")
cs1_true_sumsqfit.data <- t(apply(cs1_results_wide[,c('true_Dprime_5', 'true_Dprime_10', 'true_Dprime_20', 'true_Rprime_5', 'true_Rprime_10', 'true_Rprime_20')], 
  1, sumsqequationsolve, (cs1_results_wide$transcription_rate[1]*1000)))
cs1_results_wide$true_hl = cs1_true_sumsqfit.data[,1]

cs2_true_sumsqfit.data <- t(apply(cs2_results_wide[,c('true_Dprime_5', 'true_Dprime_10', 'true_Dprime_20', 'true_Rprime_5', 'true_Rprime_10', 'true_Rprime_20')], 
  1, sumsqequationsolve, (cs2_results_wide$transcription_rate[1]*1000)))
cs2_results_wide$true_hl = cs2_true_sumsqfit.data[,1]

print("Calculting true cleavage rates")
cs1_raw_sumsqfit.data <- t(apply(cs1_results_wide[,c('raw_Dprime_5', 'raw_Dprime_10', 'raw_Dprime_20', 'raw_Rprime_5', 'raw_Rprime_10', 'raw_Rprime_20')], 
  1, sumsqequationsolve, (cs1_results_wide$transcription_rate[1]*1000)))
cs1_results_wide$raw_hl = cs1_raw_sumsqfit.data[,1]

cs2_raw_sumsqfit.data <- t(apply(cs2_results_wide[,c('raw_Dprime_5', 'raw_Dprime_10', 'raw_Dprime_20', 'raw_Rprime_5', 'raw_Rprime_10', 'raw_Rprime_20')], 
  1, sumsqequationsolve, (cs2_results_wide$transcription_rate[1]*1000)))
cs2_results_wide$raw_hl = cs2_raw_sumsqfit.data[,1]

print("Calculating simulated cleavage rates")
cs1_count_sumsqfit.data <- t(apply(cs1_results_wide[,c('count_Dprime_5', 'count_Dprime_10', 'count_Dprime_20', 'count_Rprime_5', 'count_Rprime_10', 'count_Rprime_20')], 
  1, sumsqequationsolve, cs1_results_wide$transcription_rate[1]*1000))
cs1_results_wide$count_hl = cs1_count_sumsqfit.data[,1]

cs2_count_sumsqfit.data <- t(apply(cs2_results_wide[,c('count_Dprime_5', 'count_Dprime_10', 'count_Dprime_20', 'count_Rprime_5', 'count_Rprime_10', 'count_Rprime_20')], 
  1, sumsqequationsolve, cs2_results_wide$transcription_rate[1]*1000))
cs2_results_wide$count_hl = cs2_count_sumsqfit.data[,1]

# combine cs1 and cs2 back together
newnames <- c("true_uncleaved_5", "true_uncleaved_10", "true_uncleaved_20", 
              "true_total_5", "true_uncleaved_10", "true_uncleaved_20", 
              "true_beforeCS_5", "true_uncleaved_10", "true_uncleaved_20",   
              "true_cleaved_5", "true_cleaved_10", "true_cleaved_20", 
              "matureONLY_total_5", "matureONLY_total_10", "matureONLY_total_20",
              "matureONLY_uncleaved_5", "matureONLY_uncleaved_10", "matureONLY_uncleaved_20",
              "uncleavedONLY_total_5", "uncleavedONLY_total_10", "uncleavedONLY_total_20",
              "uncleavedONLY_uncleaved_5", "uncleavedONLY_uncleaved_10", "uncleavedONLY_uncleaved_20",
              "raw_total_5", "raw_total_10", "raw_total_20",
              "raw_uncleaved_5", "raw_uncleaved_10", "raw_uncleaved_20",
              "count_uncleaved_5", "count_uncleaved_10", "count_uncleaved_20", 
              "count_cleaved_5", "count_cleaved_10", "count_cleaved_20", 
              "count_beforeCS_5","count_beforeCS_10","count_beforeCS_20",
              "count_cleaved_5", "count_cleaved_10", "count_cleaved_20",
              "true_ratio_5", "true_ratio_10", "true_ratio_20",
              "raw_ratio_5", "raw_ratio_10", "raw_ratio_20",
              "count_ratio_5", "count_ratio_10", "count_ratio_20",
              "true_Dprime_5", "true_Dprime_10", "true_Dprime_20",
              "raw_Dprime_5", "raw_Dprime_10", "raw_Dprime_20", 
              "count_Dprime_5", "count_Dprime_10", "count_Dprime_20",
              "true_Rprime_5", "true_Rprime_10", "true_Rprime_20",
              "raw_Rprime_5", "raw_Rprime_10", "raw_Rprime_20",
              "count_Rprime_5", "count_Rprime_10", "count_Rprime_20")

colnames(cs1_results_wide) <- c("U_dist", "RTR", "TPM", "n_millions", "readlength", "mean_insert", "sd_insert",
                                "transcription_rate", "labeling", "PAU", "extUTR", "cs1_half_life",
                                paste0("cs1_", newnames), "cs1_true_hl", "cs1_raw_hl", "cs1_count_hl")
colnames(cs2_results_wide) <- c("U_dist", "RTR", "TPM", "n_millions", "readlength", "mean_insert", "sd_insert",
                                "transcription_rate", "labeling", "PAU", "extUTR", "cs2_half_life",
                                paste0("cs2_", newnames), "cs2_true_hl", "cs2_raw_hl", "cs2_count_hl")

results_wide_all <- cbind(cs1_results_wide, cs2_results_wide[,-c(1:11)])

print("Writing out half lives")
juncfile = paste0(DIR,"/twojuncSIM_042723_oldsolvePRINT_U",U_dist,"_RTR",RTR,"_X",expression_level,"_PAU",PAU,"_extUTR",extUTR,".txt")
write.table(results_wide_all, file=juncfile, sep="\t", quote=F, row.names=F, col.names=T)

print("All done!")

