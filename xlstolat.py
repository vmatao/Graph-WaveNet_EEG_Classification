import pandas as pd
df = pd.read_excel("Book1.xlsx")
df.to_latex("output.tex")