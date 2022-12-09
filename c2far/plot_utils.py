"""Generic plotting utilities.

License:
    MIT License

    Copyright (c) 2022 HUAWEI CLOUD

"""
# c2far/plot_utils.py

import matplotlib.pyplot as plt


class PlotUtils():
    """Generic utilities related to plotting."""

    @staticmethod
    def make_pdf_page(text, pdf):
        """Add a page to the PDF that just contains the given text - it's like
        a header page.

        Args:

        text: String - the info to put on the page.

        pdf: PdfPages - the pdf to save the figure in.

        """
        fig = plt.figure()
        plt.axis([0, 60, 0, 60])
        plt.axis('off')
        plt.text(57, 59, text, fontsize=6, family='monospace',
                 ha='right', va='top', wrap=True)
        pdf.savefig(fig, dpi=80)
        plt.close()
