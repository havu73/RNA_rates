import pandas as pd
import numpy as np
import os
import argparse
import helper
from scipy.special import gamma
import simulate_reads_from_transcripts as reads
READ_LENGTH = 50
TOSS_DISTANCE = 300
BUFFER_JUNCTION=9  # number of bf before the intron start site that we will consider a read to be a junction read.
# for a read length READ_LENGTH to be considered a junction read, the start of the read must be within (READ_LENGTH-BUFFER_JUNCTION-1, BUFFER_JUNCTION) bp of ahead the intron start site
DEFAULT_SEED = 9999
def set_seed(seed=DEFAULT_SEED):
    """
    Set the seed for the random number generator
    :param seed: seed for the random number generator
    :return: None
    """
    np.random.seed(seed)

def determine_junction_from_start(Udist, row, read_length=READ_LENGTH, buffer=BUFFER_JUNCTION):
    """
    This function will determine whether a read, given its start 0-based coordinate, is a junction read or not
    :param Udist: Distance from TSS to the first intron (there can be an exon before this intron)
    :param row: a row from the fragment_df, with field "start": 0-based coordinate of the read (or, the fragment, bc reads are consider to be the first READ_LENGTH bp of the fragment), "spliced": whethe the fragment is generated from a spliced transcript or not
    :param read_length: READ_LENGTH of each read based on sequencing technology
    :param buffer: BUFFER_JUNCTION, number of bf before the intron start site that we will consider a read to be a junction read.
    :return: True (junction) or False (not a junction read)
    """
    junction = (row['spliced']) & (row['start']>=Udist-read_length+buffer-1) & (row['start']<Udist-buffer)
    return junction


def write_cigar_for_read(row, Udist, Ilen, read_length=READ_LENGTH):
    """
    :param row: a row in the fragment_df that is being formatted in the .sam format
    :param Udist: Distance from TSS to the first intron (there can be an exon before this intron)
    :param Ilen: length of the intron
    :param read_length: READ_LENGTH of each read based on sequencing technology
    :return:
    the cigar string for this read (in this row) based on the splicing and the junction status of the fragment
    """
    if row['spliced'] and row['junction']:
        cigar = '{}M{}N{}M'.format(Udist-row['start'], Ilen, read_length-(Udist-row['start']))
        # first M bp are matched/unmatched, next N bp are skipped, last M bp are matched/unmatched
    cigar = '{}M'.format(read_length) # all the reads are matched/unmatched
    return cigar
"""
Ultimately, the output of this should be a .sam file with the following columns:
QNAME, FLAG, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL, TAGS
QNAME: read name
FLAG: bitwise flag, 0 for forward strand, 16 for reverse strand. It's an integer value that encodes various properties of the read and its alignment. For example, whether the read is paired, mapped, its mate is mapped, the read is unmapped, etc.
RNAME: reference sequence name (it can be gene name, chromosome name, etc.). In this case, this column is not important for the purposes of simulation, so it can be anything
POS: 1-based leftmost mapping position of the read relative to the reference sequence
MAPQ: mapping quality, Phred-scaled probability that the alignment is wrong
CIGAR: CIGAR string, describes the alignment of the read to the reference sequence. Operations like match, insertion, deletion, etc., are encoded in this string. If the read is unmapped, the CIGAR string is *.
RNEXT: Reference name of the mate/next read. It's the RNAME of the mate/next read in the template. For a single-segment template, it is *. I don't know what this field represents, really, it's not important for the purposes of simulation
PNEXT: Position of the mate/next read. It's the 1-based leftmost position of the mate/next read in the template. For a single-segment template, it is 0. 
TLEN: signed observed template length. It equals the sum of lengths of the sequenced segments and the lengths of the gaps between them. It can be positive, negative, or zero. I don't know what this field represents, really, it's not important for the purposes of simulation
SEQ: segment sequence. 
QUAL: ASCII of Phred-scaled base quality + 33. If the read is unmapped, the QUAL is *.
TAGS: additional information about the read. It's a string of the form TAG:TYPE:VALUE. For example, NH:i:1 means that the read has 1 alignment. 
"""
def format_reads_in_sam(fragment_df, Ilen, Elen, Udist, Ddist, label_time, halflife, expression, num_total_transcript_millions, read_length = READ_LENGTH):
    """
    :param fragment_df: dataframe of the current framents. columns: transcript, start, length, spliced, junction
    :param Ilen: length of the intron
    :param Elen: length of the exon that follows the intron
    :param Udist: Distance from TSS to the first intron (there can be an exon before this intron)
    :param Ddist: distance from the end of the exon to the polyA site
    :param label_time: time point at which the mRNA molecule is exposed to the label
    :param halflife: half life of P(the mRNA molecule is spliced) in minutes
    :param expression: expression level of the gene, in TPM
    :param num_total_transcript_millions: number of millions of transcripts that this cell/cells generated through transcription
    :param read_length: read_length. In this simulation, we create the reads that are the first read_length basepairs of the fragment
    :return:
    a dataframe in the .sam format. columns: read_name, flag, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL, TAG12345, CIGAR, transcript, start, length, spliced, junction
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
    # now, we will change the CIGAR string for reads that are junction reads from a spliced transcript
    fragment_df['CIGAR'] = fragment_df.apply(lambda row: write_cigar_for_read(row, Udist, Ilen, read_length), axis=1)
    return fragment_df
def simulate_transcripts(Udist, Elen, Ilen, Ddist, label_time, halflife, transcription_rate, expression, num_total_transcript_millions):
    """
    :param Udist: Distance from TSS to the first intron (there can be an exon before this intron)
    :param Elen: length of the exon that follows the intron
    :param Ilen: length of the intron
    :param Ddist: distance from the end of the exon to the polyA site
    :param label_time: time point at which the mRNA molecule is exposed to the label
    :param halflife: half life of P(the mRNA molecule is spliced) in minutes
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
    # - the transcription rate is constant across the gene
    # - All the transcripts that we generated are at least as far as the beginning of the intron
    max_transcript_length = Udist + Ilen + Elen + Ddist + label_time*transcription_rate
    # transcript is usually only as far as after Ddist, but in simulation scenario we assume that they start at the PAS and go as far as PAS+halflife*transcription_rate
    max_length_from_I = max_transcript_length - Udist
    # the maximum length of the transcript from the beginning of the intron
    endsites = Udist + np.random.randint(0, max_length_from_I, num_transcript)
    # randomly generate the end sites of the transcripts
    spliced_prob = 1-2**(-(endsites-Udist-Ilen)/(halflife*transcription_rate))
    # for each transcript, the probability that has been spliced given its end site and the splicing half life
    spliced_prob = np.where(spliced_prob < 0, 0, spliced_prob)
    # if the probability is negative (endsites before the intron ends), set it to 0 --> cannot be spliced yet
    spliced = (np.random.binomial(1, spliced_prob)==1) & (endsites>(Udist+Ilen))
    # for each transcript, whether it has been spliced or not, (num_transript,)
    transcript_lengths = endsites - Ilen*spliced
    # find the length of each transcript based on whether it has been spliced or not
    fragment_df = reads.generate_fragments(transcript_lengths)  # generate fragments from each transcript
    # a dataframe with columns ['transcript', 'start', 'length']
    # start is 0-based, counted from beginning of the transcript
    spliced = pd.Series(spliced).rename('spliced')  # name of the series, this is helpful here when we merge the series into a dataframe
    fragment_df = fragment_df.merge(spliced, left_on='transcript', right_index=True)  # a new column: 'spliced' ---> whether this fragment is from a spliced transcript or not
    fragment_df = fragment_df[fragment_df['start']>=TOSS_DISTANCE]  # toss fragments that are too close to the beginning of the transcript, because ultimately we do not really care about these transcripts, we more likely care for those that are closer to the splice site, being informative about the splicing status of the transcript
    # next, we will change the start point of fragment depending on whether the transcript is spliced or not. If spliced and the fragment overlap the intron, then the start point will be the end of the intron
    fragment_df['start'] = np.where((fragment_df['spliced']) & (fragment_df['start']>=Udist) & (fragment_df['start']<Udist+Ilen), Udist+Ilen, fragment_df['start'])
    # next, we will determine whether the reads are junction reads or not.
    # Because we will only select reads that are the first 50 bp of the fragment, for each fragment, we will consider a read generated from this fragment to be a junction read if the start of the fragment is within 9bp or abs(-readlength+10) of the intron start site
    fragment_df['junction'] = fragment_df.apply(lambda row: determine_junction_from_start(Udist, row), axis=1)
    # next, we will report the reads into a .sam format dataframe
    fragment_df = format_reads_in_sam(fragment_df, Ilen, Elen, Udist, Ddist, label_time, halflife, expression, num_total_transcript_millions)  # a dataframe in the .sam format, columns: read_name, flag, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL, TAG12345, CIGAR, transcript, start, length, spliced, junction
    return fragment_df

def write_to_sam_file(fragment_df, output_fn, Udist, Ilen, Elen, Dist, label_time, transciption_rate):
    """
    this function will write a quite-standard sam file that we can use to draw plots of coverage of reads
    :param fragment_df: output from simulate_transcripts, columns: read_name, flag, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL, TAG12345, CIGAR, transcript, start, length, spliced, junction
    :param output_fn: .sam file for which we can write the output to
    :param Udist: Distance from TSS to the first intron (there can be an exon before this intron)
    :param Ilen: length of the intron
    :param Elen: length of the exon that follows the intron
    :param Dist: distance from the end of the exon to the polyA site
    :param label_time: time point at which the mRNA molecule is exposed to the label
    :param transciption_rate: transcription rate in nt/min, right now assumed constant across the gene
    :return: None
    """
    fragment_df = fragment_df[['read_name', 'flag', 'RNAME', 'POS', 'MAPQ', 'CIGAR', 'RNEXT', 'PNEXT', 'TLEN', 'SEQ', 'QUAL', 'TAG1', 'TAG2', 'TAG3', 'TAG4', 'TAG5', 'transcript', 'start', 'length', 'spliced', 'junction']]
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
    parser.add_argument('--halflife', default=5, required=False, type=int, help='half life of P(the mRNA molecule is spliced) in minutes') # this is the half life of the splicing event, not the half life of the mRNA molecule
    parser.add_argument('--output_fn', type=str, required=True, help='output file name')
    args = parser.parse_args()
    helper.create_folder_for_file(args.output_fn)
    print('Done getting input arguments')
    fragment_df = simulate_transcripts(args.Udist, args.Elen, args.Ilen, args.Ddist, args.label_time, args.halflife, args.transcription_rate, args.expression, args.num_total_transcript_millions)
    # a dataframe of fragments from transcripts
    # columns: transcript, start, length, spliced, junction, read_name, flag, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL, TAG12345, CIGAR
    print('Done simulating transcripts')
    write_to_sam_file(fragment_df, args.output_fn, args.Udist, args.Ilen, args.Elen, args.Ddist, args.label_time, args.transcription_rate)
    print('Done writing to file')

# first, I create script that will do the splicing rate simulation given different parameters
# second, I will create script that will take in the simulated reads in .sam file and visualize the data into a coverage plot, and also script that plots the number of junction reads in the data
# third, I will create script to do simulation on the elongation rate and create script to visualize the results
# fourth, I will create script to do simulation on the cleavage rate and create script to visualize the results
# fifth, I will write a documentation on how to simulate data that combines all of these different subprocesses of transcription