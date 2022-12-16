import pandas as pd


class IntervalAnnotation:

    def __init__(self, annotation_df):
        self.df = annotation_df
        assert all(c in self.df.columns for c in ["chrom", "chromStart", "chromEnd"])

    @classmethod
    def from_bed(cls, bed_file, extra_cols):
        """Ref: https://en.wikipedia.org/wiki/BED_(file_format)"""
        df = pd.read_csv(bed_file, sep="\t", names=["chrom", "chromStart", "chromEnd"]+extra_cols)
        return cls(df)

    @classmethod
    def from_gap(cls, gap_file):
        """Ref: https://genome.ucsc.edu/cgi-bin/hgTables?db=hg19&hgta_group=map&hgta_track=gap&hgta_table=gap&hgta_doSchema=describe+table+schema"""
        df = pd.read_csv(gap_file, sep='\t',
                         names=["ucscBin", "chrom", "chromStart", "chromEnd",
                                "ix", "n", "size", "type", "bridge"])
        return cls(df)

    def get_chrom_annotated_bins(self, chrom, bin_size=25):
        annotated_bins = []
        chrom_annotated = self.df[self.df["chrom"]==chrom]
        print(f"Number of annotated intervals in {chrom}:  {len(chrom_annotated)}")

        for ix, row in chrom_annotated.iterrows():
            start_bin = row["chromStart"] // bin_size  # the bin in which the annotation starts
            end_bin = row["chromEnd"] // bin_size  # the bin preceding the one in which the annotation ends
            # make correction for when row['end'] coincides with a bin end. N.B. this is new behaviour.
            if row["chromEnd"] % bin_size > 0:
                end_bin += 1
            annotated_bins += [b for b in range(start_bin, end_bin)]
        
        return annotated_bins
