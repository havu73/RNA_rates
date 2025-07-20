import pandas as pd
import random
import argparse
import math
import os


# Create the DNA sequence from the dataframe
def getDNAseq(dna_sequence):
    return ''.join(str(i) for i in dna_sequence)

# Find the stop site (end of labeled region)
#findStop(df, 0.72, 1)

def findStop(df, labelingtime, startsite):
    # Find in which feature the start site is located
    tmp = df.loc[(df.start <= startsite) & (df.end >= startsite)]
    # Get the index of the feature in which the start site is located
    idx = tmp.index.values[0]
    #rate=statistics.mean(df.txrate)
    #variability=random.uniform(0.9, 1.1) #10% variability in transcription rate for each read
    variability=1
    timeleft = labelingtime - (df.end[idx] - startsite) / (df.txrate[idx] * 1000 * variability)
    while timeleft > 0:
        # Move on to the next feature if there is still time left
        idx += 1
        # If there is time left, subtract the time it takes to transcribe the next feature
        try:
            timeleft += -df.time[idx]
        # If we reached the last feature in the gene sequence, then the stop site is the end of the gene sequence
        except KeyError:
            return int(df.end[idx - 1])
    # If the end of the gene sequence has not been reached in the given labeling time, we must find where transcription has stopped
    else:
        # Find the txrate for the next feature that is entered
        ntpertime = (df.txrate[idx] * 1000 * variability)  #10% variability in transcription rate
        # Find the number of nt that will be transcribed, based on the txrate and the time left from the labeling period
        nts = abs(timeleft * ntpertime)
        # Subtract the number of nt from the end of the sequence to get the stop site
        stop = df.end[idx] - math.ceil(nts)
        return stop

# Perform U to U conversion within the labeled region
def convert_base(read, percent_range, start_site, stop_site, from_base, to_base):
    percent = random.randint(percent_range[0], percent_range[1])  # percent of bases to convert
    labeled_region = read[start_site:stop_site]
    pos = [pos for pos, char in enumerate(labeled_region) if char == from_base]
    num_bases_conv = int(math.ceil(len(pos) * percent / 100))
    pos_conv = random.sample(pos, num_bases_conv)
    conv_read = list(read)
    for i in pos_conv:
        conv_read[start_site + i] = to_base
    return ''.join(conv_read), percent, [start_site + i for i in pos_conv]

def mutate_bg(read, percent_range):
    read_length = len(read)
    percent = round(random.uniform(percent_range[0], percent_range[1]), 1)
    num_bkgd_mut = math.ceil(read_length * percent / 100)
    pos = [pos for pos, char in enumerate(read)]
    pos_mut = random.sample(pos, int(num_bkgd_mut))
    pos_mut.sort()
    mut_read = list(read)
    for i in pos_mut:
        rand_nt = random.choice(["A", "C", "G", "T"])
        while (rand_nt == mut_read[i]):
            rand_nt = random.choice(["A", "C", "G", "T"])
        ref_nt = mut_read[i]
        mut_nt = rand_nt
        mut_read[i] = mut_nt
    # Convert the list back into a string
    mut_read = "".join(mut_read)

    # return mut_read, pos_mut, bkgd_mut
    return mut_read, percent, pos_mut

def createOutputName(gtf_file, label_time, bkgd_perc_range, u2c_perc_range, num_reads):
    gtf_name = gtf_file.strip(".gtf")
    output_file_name = str(gtf_name) + "_l" + str(label_time) + "_b" + str(bkgd_perc_range) + "_p" + str(
        u2c_perc_range) + "_reads" + str(num_reads) + ".csv.gz"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--f', default='tmp.gtf', help='Input GTF file name')
    parser.add_argument('--l', type=int, default=5, help='Labeling time in minutes')
    parser.add_argument('--b', default='0.1,0.2', help='Background mutation rate range')
    parser.add_argument('--subs_5eU', type=str, default='5,15', help='range of percent conversion for 5eU', required=False)
    parser.add_argument('--subs_6sG', type=str, default='5,15', help='range of percent conversion for 6sG', required=False)
    parser.add_argument('--subs_4sU', type=str, default='5,15', help='range of percent conversion for 4sU', required=False)
    parser.add_argument('--reads', type=int, default=100, help='Number of reads to generate')
    parser.add_argument('--o', type=str, default='./', help='output path')
    args = parser.parse_args()

    u2u_percent_range = list(map(lambda x: int(x), args.subs_5eU.split(',')))
    g2a_percent_range = list(map(lambda x: int(x), args.subs_6sG.split(',')))  # *** ORIGINAL ***
    u2c_percent_range = list(map(lambda x: int(x), args.subs_4sU.split(',')))
    seqerr_range = list(map(lambda x: float(x), args.b.split(',')))

    u2u_subrate_name = str(u2u_percent_range[0]) + '-' + str(u2u_percent_range[1])
    g2a_subrate_name = str(g2a_percent_range[0]) + '-' + str(g2a_percent_range[1])
    u2c_subrate_name = str(u2c_percent_range[0]) + '-' + str(u2c_percent_range[1])
    seqerr_name = str(seqerr_range[0]) + '-' + str(seqerr_range[1])

    df = pd.read_csv(args.f, sep='\t', comment='#') # this is a  file showing the gene structure: exons, introns, and the sequences of each part
    df.columns = ['chromosome','source','feature','start','end','length','txrate','time','sequence']

    dna_sequence = getDNAseq(df.sequence)  # a string combining the sequence of all the introns and exons in the gene
    
    #Generate output name and file
    filename = os.path.splitext(os.path.basename(args.f))[0]
    #output_filename = str(args.o)+filename + '_p' + str(args.p) + '_ptime' + str(args.l) + '_t'

    output_filename = str(args.o)+filename + '_5eUrates_' + str(u2u_subrate_name) + '_6sGrates_' + str(g2a_subrate_name) + '_4sUrates_' + str(u2c_subrate_name) + '_ptime_' + str(args.l) + '_seqerr_' + str(seqerr_name) + '_reads_' + str(args.reads) + '.tsv'

    output_file = open(output_filename, 'w')

    #Generate initiation times
    initiation_times=[]
    for i in range(0,args.reads):
        initiation_temp=random.uniform(0,args.l*3) #I'm actually using this as TIME LEFT IN THE EXPERIMENT
        initiation_times.append(initiation_temp)
    n=0

    data_list = []

    for _ in range(args.reads):
        test = initiation_times[n]  # shouldn't this ben initiation_times[n]?  and for n in range(args.reads)?
        initiation=test
        starts_5eU_pos=1  #
        stop_site_5eU = findStop(df, initiation, starts_5eU_pos) # initiation is when labeling starts
        if initiation>args.l*2:#If there are more than 2timepoints mins (meaning that there's only 5eU available)
            starts_6sG_pos = findStop(df, initiation-args.l*2, 1)
            #print(initiation)
            #print(args.l*2)
            #print(starts_6sG_pos)
            stop_site_6sG = findStop(df, args.l*2, starts_6sG_pos)#6sG will be available for the entire 2+3 timepoints
        else:
            starts_6sG_pos = 1
            stop_site_6sG = findStop(df, initiation, starts_6sG_pos)
        if initiation>args.l:
            starts_4sU_pos = findStop(df, initiation-args.l, 1)
            stop_site_4sU = findStop(df, args.l, starts_4sU_pos)#4sU will be available for the entire 3 timepoint
        else:
            starts_4sU_pos = 1
            stop_site_4sU = findStop(df, initiation, starts_4sU_pos)

        read = dna_sequence[starts_5eU_pos:int(stop_site_5eU)]
        converted_read, percent_u2u, converted_positions_5eU = convert_base(read, u2u_percent_range, int(starts_5eU_pos), int(stop_site_5eU),'T','T')
        converted_read, percent_g2a, converted_positions_6sG = convert_base(converted_read, g2a_percent_range, int(starts_6sG_pos), int(stop_site_6sG),'G','A')
        converted_read, percent_t2c, converted_positions_4sU = convert_base(converted_read,u2c_percent_range , int(starts_4sU_pos), int(stop_site_4sU),'T','C')

        #bg_mutated_read, percentage_bg_mutations, mutated_bg_positions = mutate_bg(converted_read, list(map(float, args.b.split(','))), int(starts_5eU_pos), int(stop_site_5eU))
        bg_mutated_read, percentage_bg_mutations, mutated_bg_positions = mutate_bg(converted_read, list(map(float, args.b.split(','))))
        n = n + 1
        row_data = [initiation,percent_u2u, percent_g2a, percent_t2c,starts_5eU_pos, stop_site_5eU, starts_6sG_pos, stop_site_6sG,starts_4sU_pos, stop_site_4sU,percentage_bg_mutations, converted_positions_5eU,converted_positions_6sG,converted_positions_4sU,mutated_bg_positions,bg_mutated_read]
        # Add the row_data to the data_list

        data_list.append(row_data)

    with open(output_filename, 'w') as output_file:
        # ha's code added to this, but let's not mess with it just yet
        # colnames = ['initiation', 'percent_u2u', ' percent_g2a', ' percent_t2c', 'starts_5eU_pos', ' stop_site_5eU', ' starts_6sG_pos', ' stop_site_6sG', 'starts_4sU_pos', ' stop_site_4sU', 'percentage_bg_mutations', ' converted_positions_5eU', 'converted_positions_6sG', 'converted_positions_4sU', 'mutated_bg_positions', 'bg_mutated_read']
        # output_file.write('\t'.join(colnames) + '\n')
        for row_data in data_list:
            output_file.write('\t'.join(map(str, row_data)) + '\n') 
