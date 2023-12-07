"""Generic plotting utilities.

License:
    MIT License

    Copyright (c) 2023 HUAWEI CLOUD

"""
# sutranets/plot_utils.py
import matplotlib.pyplot as plt


class PlotUtils():
    __DPI = 80

    @classmethod
    def make_pdf_page(cls, text, pdf):
        """Add a header page to the PDF that just contains the given text.

        """
        fig = plt.figure()
        plt.axis([0, 60, 0, 60])
        plt.axis('off')
        plt.text(57, 59, text, fontsize=6, family='monospace',
                 ha='right', va='top', wrap=True)
        pdf.savefig(fig, dpi=cls.__DPI)
        plt.close()
