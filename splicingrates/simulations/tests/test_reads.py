# TODO: this file is not up-to-date.
# I need to update it to reflect the current state of the code after the project finishes

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # this is supposed to let the code know that some of the import path is outside of this folder
import unittest
import pandas as pd
from transcription.simulate_reads_from_transcripts import check_intron_break_per_read, adjust_absolute_one_read_coords  # Replace 'your_module' with the name of your module
from pandas.testing import assert_series_equal
from transcription.transcripts import Transcript

class CustomTest(unittest.TestCase):
    def assertSeriesEqual(self, series1, series2, msg=None):
        try:
            assert_series_equal(series1, series2)
        except AssertionError as e:
            raise self.failureException(msg or str(e))

class TestCheckIntronBreakFunction(CustomTest):
    def __init__(self, methodName='check_intron_break_per_read', param=None):
        super().__init__(methodName)
        self.trans_spl = pd.DataFrame({'txrate': 0, 'start': [10, 30, 50], 'is_spliced': [1, 0, 1], 'Ilen': 10, 'trans_idx': 0, 'end': [20, 40, 60]})
        self.trans_spl['Udist'] = self.trans_spl['start']
        self.trans_spl['spliced_out_len'] = self.trans_spl['Ilen'] * self.trans_spl['is_spliced']
        self.trans_spl['Blen'] = self.trans_spl['spliced_out_len'].transform(lambda x: x.cumsum().shift(fill_value=0))
        self.trans_spl['Alen'] = self.trans_spl['Blen'] + self.trans_spl['spliced_out_len']
        self.gene_start=0
        self.transcript_length=70  # E1, I1, E2, I2, E3, I3, E4 --> each exon is 10 bp, each intron is 10 bp
        self.trans_spl['start_within_trans'] = self.trans_spl['start'] - self.gene_start - self.trans_spl['Blen']
        self.N = self.trans_spl.shape[0] # number of introns in this transcript
        spl_int_start_list = self.trans_spl.start_within_trans.tolist()
        # the first N introns are the ones that are spliced out. Of course this is only true when we do not simulate alternative splicing
        self.intron_read_start= [0]+spl_int_start_list+[self.transcript_length]
        self.is_spliced = [0] + self.trans_spl.is_spliced.tolist() + [0] # first and last is not spliced out because it corresponds to the start and end of the transcript, not an intron
    def test_before_I1(self):
        row = pd.Series({'rel_start': 0, 'rel_end': 5})
        right_answer = pd.Series({'overlap_splicedI': False, 'precedeI': 0})
        pd.testing.assert_series_equal(check_intron_break_per_read(row, self.intron_read_start, self.is_spliced), right_answer)

    def test_overlap_I1(self):
        row = pd.Series({'rel_start': 8, 'rel_end': 13})
        right_answer = pd.Series({'overlap_splicedI': True, 'precedeI': 0})
        pd.testing.assert_series_equal(check_intron_break_per_read(row, self.intron_read_start, self.is_spliced), right_answer)

    def test_after_I1(self):
        row = pd.Series({'rel_start': 15, 'rel_end': 20})
        right_answer = pd.Series({'overlap_splicedI': False, 'precedeI': 1})
        pd.testing.assert_series_equal(check_intron_break_per_read(row, self.intron_read_start, self.is_spliced), right_answer)

    def test_overlap_I2(self):  # I2 is not spliced out, so this should return False
        row = pd.Series({'rel_start': 16, 'rel_end': 21})
        right_answer = pd.Series({'overlap_splicedI': False, 'precedeI': 1})
        pd.testing.assert_series_equal(check_intron_break_per_read(row, self.intron_read_start, self.is_spliced), right_answer)
        row = pd.Series({'rel_start': 22, 'rel_end': 27})
        right_answer = pd.Series({'overlap_splicedI': False, 'precedeI': 2})
        pd.testing.assert_series_equal(check_intron_break_per_read(row, self.intron_read_start, self.is_spliced), right_answer)

    def test_after_I2(self): # read after I2 but before I3
        row = pd.Series({'rel_start': 34, 'rel_end': 39})
        right_answer = pd.Series({'overlap_splicedI': False, 'precedeI': 2})
        pd.testing.assert_series_equal(check_intron_break_per_read(row, self.intron_read_start, self.is_spliced), right_answer)

    def test_overlap_I3(self): # I3 is spliced out
        row = pd.Series({'rel_start': 35, 'rel_end': 40}) # read starts right before I3
        right_answer = pd.Series({'overlap_splicedI': False, 'precedeI': 2})
        pd.testing.assert_series_equal(check_intron_break_per_read(row, self.intron_read_start, self.is_spliced), right_answer)
        row = pd.Series({'rel_start': 36, 'rel_end': 41}) # reads overlaps with I3 which is spliced out
        right_answer = pd.Series({'overlap_splicedI': True, 'precedeI': 2})
        pd.testing.assert_series_equal(check_intron_break_per_read(row, self.intron_read_start, self.is_spliced), right_answer)

    def test_after_I3(self):
        row = pd.Series({'rel_start': 42, 'rel_end': 47})
        right_answer = pd.Series({'overlap_splicedI': False, 'precedeI': 3})
        pd.testing.assert_series_equal(check_intron_break_per_read(row, self.intron_read_start, self.is_spliced), right_answer)

    def test_adj_reads(self):
        """
        Test the function that adjusts the reads' coordinates by the spliced out length of the introns
        :return:
        """
        reads_in_trans = pd.DataFrame({'rel_start': [0, 8, 15, 16, 22, 34, 35, 36, 42],
                                       'rel_end':   [5, 13, 20, 21, 27, 39, 40, 41, 47],
                                       'trans_idx': 0})
        reads_in_trans[['overlap_splicedI', 'precedeI']] = reads_in_trans.apply(lambda row: check_intron_break_per_read(row, self.intron_read_start, self.is_spliced), axis=1)
        abs_reads = pd.DataFrame({'abs_start':      [0, 8,  25, 26, 32, 44, 45, 46, 62],
                                  'abs_end':        [5, 23, 30, 31, 37, 49, 50, 61, 67],
                                  'subtract_start': [0, 10,  0, 0,  0,  0,  0, 50,  0],
                                  'subtract_end':   [0, 20,  0, 0,  0,  0,  0, 60,  0],
                                  'trans_idx': 0,
                                  'precedeI':        [0, 0,  1, 1,  2,  2,  2,  2, 3]})
        self.splicedO_b4_reads = self.trans_spl['spliced_out_len']+self.trans_spl['is_spliced']*self.trans_spl['Ilen']
        self.splicedO_b4_reads = [0] + list(self.splicedO_b4_reads) + [self.splicedO_b4_reads.iloc[-1]]
        # if a read is assigned start after each intron, then this is the length of the regions that have been spliced out before this read (taking into account whether this intron has been spliced out or not). If the read starts after a spliced intron, the intron will be counted into spliced out region. If the read starts after the start of a unspliced intron, the intron will not be counted into spliced out region.
        Transcript.gene_start=0 # method attribute
        transcript = Transcript(0, self.transcript_length, self.trans_spl, reads_in_trans)
        transcript._set_length()
        transcript._map_reads_to_gene()
        self.assertSeriesEqual(transcript.reads_df.abs_start, abs_reads.abs_start)
        self.assertSeriesEqual(transcript.reads_df.abs_end, abs_reads.abs_end)



if __name__ == '__main__':
    unittest.main()
