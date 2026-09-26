import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl

s1 = pl.read_csv('student_resource/dataset/test/test_source1.tsv', separator='\t', 
    schema_overrides={'entity_id': pl.Utf8, 'business_name': pl.Utf8, 'business_address': pl.Utf8, 'country': pl.Utf8})

# Check distribution of first word
s1 = s1.with_columns(
    pl.col('business_name').str.to_lowercase().str.extract(r'^(\w+)', 1).alias('first_word')
)

# Top 20 first words
top = s1.group_by('first_word').agg(pl.len().alias('cnt')).sort('cnt', descending=True).head(20)
print(top)
print()
print("Unique first words:", s1["first_word"].n_unique())
print("Total rows:", len(s1))

# Check first two words
s1 = s1.with_columns(
    pl.col('business_name').str.to_lowercase().str.extract(r'^(\w+\s+\w+)', 1).fill_null(
        pl.col('business_name').str.to_lowercase().str.extract(r'^(\w+)', 1)
    ).alias('first_two')
)
top2 = s1.group_by('first_two').agg(pl.len().alias('cnt')).sort('cnt', descending=True).head(20)
print()
print("First two words distribution:")
print(top2)
print("Unique first-two:", s1["first_two"].n_unique())
