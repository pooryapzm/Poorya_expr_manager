from pylatex import Document, LongTable, MultiColumn, Section


def dataframe_to_pdf(df, pdf_path):
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)

    num_cols = len(df.columns)

    # Generate data table
    with doc.create(LongTable("l " * num_cols)) as data_table:
        data_table.add_hline()
        data_table.add_row(list(df.columns.values))
        data_table.add_hline()
        for i in range(len(df.index)):
            row_list = df.iloc[i].values.tolist()
            if "SEP" in row_list[0]:
                data_table.add_hline()
            else:
                data_table.add_row(row_list)
        data_table.add_hline()

    doc.generate_pdf(pdf_path, clean_tex=False)


def dataframe_list_to_pdf(df_list, title_list, pdf_path):
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)

    for i,df in enumerate(df_list):
        num_cols = len(df.columns)
        with doc.create(Section(title_list[i])):
            # Generate data table
            with doc.create(LongTable("l " * num_cols)) as data_table:
                data_table.add_hline()
                data_table.add_row(list(df.columns.values))
                data_table.add_hline()
                for i in range(len(df.index)):
                    row_list = df.iloc[i].values.tolist()
                    if "SEP" in row_list[0]:
                        data_table.add_hline()
                    else:
                        data_table.add_row(row_list)
                data_table.add_hline()

    doc.generate_pdf(pdf_path, clean_tex=False)

