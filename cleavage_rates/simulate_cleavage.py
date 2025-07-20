import pandas as pd
import numpy as np
import os
import argparse
import helper
from scipy.special import gamma
READ_LENGTH = 50
TOSS_DISTANCE = 300
BUFFER_JUNCTION=9  # number of bf before the intron start site that we will consider a read to be a junction read.
# for a read length READ_LENGTH to be considered a junction read, the start of the read must be within (READ_LENGTH-BUFFER_JUNCTION-1, BUFFER_JUNCTION) bp of ahead the intron start site

def calculate_breakpoints_given_transcript_length(trans_length, n, delta):
    """
    Given the length of the transcript and the number of fragments that we want to break the transcript into, this function will return a list showing the start point of each fragment in the transcript
    :param trans_length: transcript length (int)
    :param n: number of fragments (int)
    :param delta: parameters for the Wilbull distribution (float), log10(trans_length)
    :return: list of start points of each fragment (array)
    """
    relative_break = np.random.uniform(0, 1, n-1)  # number of breakpoints is number of fragments - 1
    relative_break = np.sort(relative_break)  # [0.3,0.4]
    relative_break = np.concatenate((np.array([0]), relative_break, np.array([1])))  # [0, 0.3, 0.4, 1]
    relative_length = np.diff(relative_break)  # [0.3, 0.1, 0.6]
    relative_length = np.power(relative_length, 1/delta)
    relative_length = relative_length / np.sum(relative_length)
    frag_length = np.round(trans_length * relative_length, 0)
    frag_start = np.concatenate((np.array([0]), frag_length))
    frag_start = np.cumsum(frag_start)[:-1]  # the last element is the length of the transcript, which is not the start of any fragment
    return frag_length, frag_start

def generate_fragments(trans_lengths, eta_val=200, insertsize_min=200, insertsize_max=300):
    """
    Generate fragments from each transcript, size select, and return the starting position of the resulting
    reads relative to the length of the transcript
    :param trans_lengths: the length of the transcript, a vector of size (num_transcript)
    :param eta_val: eta value input to the Weibull distrubtion
    :param insertsize_min: minimum length of selected fragement (size-select)
    :param insertsize_max: maximum length of selected fragement (size-select)
    :return: a dataframes showing the fragment start within each transcript. columns ['transcript', 'start', 'length']
    """
    num_transcript = len(trans_lengths)
    deltas = np.log10(trans_lengths)
    n = np.round(trans_lengths / (eta_val * gamma(1+ 1/trans_lengths))).astype(int)  # num_fragments, based on the formulation of  https://academic.oup.com/nar/article/40/20/10073/2414449, foundation: Weibull distribution
    n = np.where(n < 0, 0, n)  # array of number of fragments for each transcript --> (num_transcript,)
    # n stands for the number of fragments that we will break the transcript into
    breakpoint_stats_list = list(map(lambda x: calculate_breakpoints_given_transcript_length(trans_lengths[x], n[x], deltas[x]), range(num_transcript)))
    frag_length = list(map(lambda x: x[0], breakpoint_stats_list))  # list of arrays, each array is the length of each fragment in the transcript
    frag_start = list(map(lambda x: x[1], breakpoint_stats_list))  # list of arrays, each array is the start point of each fragment in the transcript, 0-based
    # apply function to calculate the fragments generated from each transcript
    # --> list of arrays. Outter list: each transcript. Inner array: length of each fragment in the transcript
    df = pd.DataFrame(columns= ['transcript', 'start', 'length'])
    transcript_indices = np.repeat(range(len(trans_lengths)), n)
    df['transcript'] = transcript_indices
    df['start'] = np.concatenate(frag_start)
    df['length'] = np.concatenate(frag_length)
    df = df[(df['length']>= insertsize_min) & (df['length']<= insertsize_max)] # size select the fragment
    return df


def format_reads_in_sam(fragment_df, Ilen, Elen, Udist, Ddist, label_time, halflife, expression, num_total_transcript_millions, read_length = READ_LENGTH):
    """
    :param fragment_df: dataframe of the current framents. columns: transcript, start, length, cleaved, junction
    :param Ilen: length of the intron
    :param Elen: length of the exon that follows the intron
    :param Udist: Distance from TSS to the first intron (there can be an exon before this intron)
    :param Ddist: distance from the end of the exon to the polyA site
    :param label_time: time point at which the mRNA molecule is exposed to the label
    :param halflife: half life of P(the mRNA molecule is cleaved) in minutes
    :param expression: expression level of the gene, in TPM
    :param num_total_transcript_millions: number of millions of transcripts that this cell/cells generated through transcription
    :param read_length: read_length. In this simulation, we create the reads that are the first read_length basepairs of the fragment
    :return:
    a dataframe in the .sam format. columns: read_name, flag, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL, TAG12345, CIGAR, transcript, start, length, cleaved, junction
    """
    fragment_df['read_name'] = 'i{Ilen}:e{Elen}:u{Udist}:d{Ddist}:L{label_time}:hl{halflife}:X{expression}:M{num_total_transcript_millions}'.format(Ilen=Ilen, Elen=Elen, Udist=Udist, Ddist=Ddist, label_time=label_time, halflife=halflife, expression=expression, num_total_transcript_millions=num_total_transcript_millions)
    fragment_df['flag'] = 0
    fragment_df['RNAME'] = 'intron{}'.format(Ilen)
    fragment_df['POS'] = fragment_df['start'] + 1  # change the start from 0-based to 1-based
    fragment_df['MAPQ'] = 50
    fragment_df['RNEXT'] = '*'
    fragment_df['PNEXT'] = 0
    fragment_df['TLEN'] = 0
    fragment_df['SEQ'] = 'A'*read_length
    fragment_df['QUAL'] = '*'  # default quality score, this means that the read is unmapped, not sure why though
    fragment_df['TAG1'] = 'XA:i:1'  # XA:i:1 means that the read has 1 alignment
    fragment_df['TAG2'] = 'MD:Z:{}'.format(read_length)  # MD:Z:50 means that all the bases in the read match the reference sequence
    fragment_df['TAG3'] = 'NM:i:0'  # NM:i:0 means that there is no mismatch in the read
    fragment_df['TAG4'] = 'NH:i:1'  # NH:i:1 means that the read has 1 alignment
    fragment_df['TAG5'] = 'XS:A:+'  # XS:A:+ means that the read is mapped to the forward strand
    # now, we will change the CIGAR string for reads that are junction reads from a cleaved transcript
    fragment_df['CIGAR'] = fragment_df.apply(lambda row: write_cigar_for_read(row, Udist, Ilen, read_length), axis=1)
    return fragment_df
def simulate_transcripts(Udist, RTR, label_time, halflife, transcription_rate, expression, num_total_transcript_millions):
    """
    This function will:
    - Simulate transcripts' possible end sites at the end of the labeling period.
    When the experiment starts, all the transcripts start from the beginning of the TSS and end somewhere, ranging from the TSS to the end of the gene. As the experiment of 4sU labeling goes on, the transcript can be elongated by Lr, so the end point can be anything from 0 (newly-initiated transcripts) to Udist+RTR+Lr (transcripts that are at Udist+RTR at the beginning of the experiment)
    In practice transcripts can only be as long as Udist+RTR, because beyond that PolII will stop transcribing.
    But for the sake of simulation, transcripts that are supposedly longer than Udist+RTR will supposedly have a higher probability of being cleaved. We will therefore simulate transcripts that can be longer than Udist+RTR to precisely calculate the probability of cleavage. But after the proability of cleavage is calculated, we will toss the end parts of transcripts that are longer than Udist+RTR.
    - Given the end sites of the transcripts, we will calculate the probability that the transcript has been cleaved given the splicing half life. P(cleaved|x) = 1-2^(-(x-Udist)/(halflife * transcription_rate)), where t0 is the time point at which the mRNA molecule is exposed to the label. x refers to the end point of the transcript.
    Why this formula? P(uncleaved|x=Udist) = 1 --> any transcripts ending before or at Udist will not be cleaved yet.
    P(uncleaved| x=Udist+halflife*transcription_rate) = 0.5 --> any transcripts ending at Udist+halflife*transcription_rate will have a 50% chance of being cleaved, because halflife is defined as the time it takes for the uncleaved transcripts to be be reduced by half, replaced by cleaved transcripts.
    - Cleave transcripts, based on the calculated probabilities
    - Generate fragments from the transcripts, based on Weibull distribution
    - For each fragment, the first READ_LENGTH bp is considered to be a read. This is actually a true reflection of the sequencing process.
    - Determine the number of informative reads about the cleaveage status of the gene. Two types of reads are considered informative: (1) reads at the junction of the cleavage site (uncleaved transcript). (2) reads a reasonable distance (N bp) from the cleavage site (values of N will be discussed later). Rationale for this is that # reads before the CS is a proxy for the number of transcripts that are both uncleaved and cleaved. # reads AT the CS is a proxy for the number of transcripts that are uncleaved.
    :param Udist: Distance from TSS to the cleavage site.
    :param RTR: length of the read through region (from the cleavage site to where the RNA Pol II stops, which is also called transcription end site)
    :param label_time: time point at which the mRNA molecule is exposed to the label
    :param halflife: half life of P(the mRNA molecule is cleaved) in minutes
    :param transcription_rate: transcription rate in nt/min, right now assumed constant across the gene
    :param num_transcript: number of transcripts to simulate
    :return:
    """
    num_transcript = expression * num_total_transcript_millions
    # the reaason why we need the total number of transcript is because:
    # - TMP_i = (RPK_i / \sum_j-1^{num_gene} RPK_j) * 10^6--> relative abundance of transcription in gene i
    # --> TPM_i/10^6 = relative abundance of transcription in gene i compared to other genes
    # - RPK_i = number of reads mapped to gene i/ length of gene i --> control for hte fact that there are more reads in a longer gene than a shorter gene
    # If we want to simulate transcripts of a gene, given its TPM, we need to know how many transcript in total were generated from all the genes, this is what num_total_transcript_millions is for.
    # N_i = TPM_i/10^6 * num_total_transcript = TPM_i * num_total_transcript_millions--> number of transcripts of gene i
    # Assumptions in this simulation:
    # - the transcription rate is constant across the gene length
    # - All the transcripts that we generated are at least as far as the beginning of the intron
    max_transcript_length = Udist + RTR + label_time*transcription_rate # maximum length possible in the simulation transcripts.
    endsites = np.random.randint(0, max_transcript_length, num_transcript) # randomly select transcript end sites. This is the end site of the transcript at the end of the labeling period.
    cleaved_prob = 1-2**(-(endsites-Udist)/(halflife*transcription_rate))
    # for each transcript, the probability that has been cleaved given its end site and the cleavage half life
    cleaved_prob = np.where(cleaved_prob < 0, 0, cleaved_prob)
    # if the probability is negative (endsites before the gene end), set it to 0 --> cannot be cleaved yet
    cleaved = (np.random.binomial(1, cleaved_prob)==1)
    # for each transcript, whether it has been cleaved or not, (num_transript,)
    transcript_lengths = endsites * (1-cleaved) + (Udist+RTR) * cleaved  # if cleaved, lengths should be Udist+RTR, if not cleaved, lengths should be endsites generated from the simulation
    fragment_df = generate_fragments(transcript_lengths)  # generate fragments from each transcript
    # a dataframe with columns ['transcript', 'start', 'length']
    # start is 0-based, counted from beginning of the transcript
    cleaved = pd.Series(cleaved).rename('cleaved')  # name of the series, this is helpful here when we merge the series into a dataframe
    fragment_df = fragment_df.merge(cleaved, left_on='transcript', right_index=True)  # a new column: 'cleaved' ---> whether this fragment is from a cleaved transcript or not
    fragment_df = fragment_df[fragment_df['start']>=TOSS_DISTANCE]  # toss fragments that are too close to the beginning of the transcript, because ultimately we do not really care about these transcripts, we more likely care for those that are closer to the cleavage site, being informative about the cleavage rate of the gene
    return fragment_df

def write_to_sam_file(fragment_df, output_fn, Udist, Ilen, Elen, Dist, label_time, transciption_rate):
    """
    this function will write a quite-standard sam file that we can use to draw plots of coverage of reads
    :param fragment_df: output from simulate_transcripts, columns: read_name, flag, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL, TAG12345, CIGAR, transcript, start, length, cleaved, junction
    :param output_fn: .sam file for which we can write the output to
    :param Udist: Distance from TSS to the first intron (there can be an exon before this intron)
    :param Ilen: length of the intron
    :param Elen: length of the exon that follows the intron
    :param Dist: distance from the end of the exon to the polyA site
    :param label_time: time point at which the mRNA molecule is exposed to the label
    :param transciption_rate: transcription rate in nt/min, right now assumed constant across the gene
    :return: None
    """
    fragment_df = fragment_df[['read_name', 'flag', 'RNAME', 'POS', 'MAPQ', 'CIGAR', 'RNEXT', 'PNEXT', 'TLEN', 'SEQ', 'QUAL', 'TAG1', 'TAG2', 'TAG3', 'TAG4', 'TAG5', 'transcript', 'start', 'length', 'cleaved', 'junction']]
    outF = open(output_fn, "w")
    max_sequence_length = Udist+ Ilen + Elen + Dist + label_time*transciption_rate+1
    outF.write('@SQ\tSN:intron{iLen}\tLN:{max_length}\n'.format(iLen=Ilen, max_length=max_sequence_length))
    fragment_df.to_csv(outF, sep='\t', header=False, index=False)
    outF.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Udist', default=500, type=int, required=False, help='Distance from TSS to the first intron (there can be an exon before this intron)')
    parser.add_argument('--Ilen', default=40, type=int, required=False, help='Length of the first intron')
    parser.add_argument('--Elen', default=300, type=int, required=False, help='Length of the exon that follows the intron')
    parser.add_argument('--Ddist', default=5000, type=int, required=False, help='distance from the end of the exon to the polyA site')
    parser.add_argument('--label_time', default=5, type=int, required=False, help='How many minutes that the mRNA molecule is exposed to the label')
    parser.add_argument('--transcription_rate', default=1500, required=False, type=float, help='Transcription rate in nt/min, right now assumed constact across the gene')
    parser.add_argument('--expression', default=5, type=int, required=False, help='expression level of the gene, in TPM')
    parser.add_argument('--num_total_transcript_millions', default=100, required=False, type=float, help='number of millions of transcripts that this cell/cells generated through transcription')
    # the reaason why we need the total number of transcript is because:
    # - TMP_i = (RPK_i / \sum_j-1^{num_gene} RPK_j) * 10^6--> relative abundance of transcription in gene i
    # --> TPM_i/10^6 = relative abundance of transcription in gene i compared to other genes
    # - RPK_i = number of reads mapped to gene i/ length of gene i --> control for hte fact that there are more reads in a longer gene than a shorter gene
    # If we want to simulate transcripts of a gene, given its TPM, we need to know how many transcript in total were generated from all the genes, this is what num_total_transcript_millions is for.
    # N_i = TPM_i/10^6 * num_total_transcript = TPM_i * num_total_transcript_millions--> number of transcripts of gene i
    parser.add_argument('--halflife', default=5, required=False, type=int, help='half life of P(the mRNA molecule is cleaved) in minutes') # this is the half life of the splicing event, not the half life of the mRNA molecule
    parser.add_argument('--output_fn', type=str, required=True, help='output file name')
    args = parser.parse_args()
    helper.create_folder_for_file(args.output_fn)
    print('Done getting input arguments')
    fragment_df = simulate_transcripts(args.Udist, args.Elen, args.Ilen, args.Ddist, args.label_time, args.halflife, args.transcription_rate, args.expression, args.num_total_transcript_millions)
    # a dataframe of fragments from transcripts
    # columns: transcript, start, length, cleaved, junction, read_name, flag, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL, TAG12345, CIGAR
    print('Done simulating transcripts')
    write_to_sam_file(fragment_df, args.output_fn, args.Udist, args.Ilen, args.Elen, args.Ddist, args.label_time, args.transcription_rate)
    print('Done writing to file')

# first, I create script that will do the splicing rate simulation given different parameters
# second, I will create script that will take in the simulated reads in .sam file and visualize the data into a coverage plot, and also script that plots the number of junction reads in the data
# third, I will create script to do simulation on the elongation rate and create script to visualize the results
# fourth, I will create script to do simulation on the cleavage rate and create script to visualize the results
# fifth, I will write a documentation on how to simulate data that combines all of these different subprocesses of transcription