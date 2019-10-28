FROM tensorflow/tensorflow:2.0.0-gpu-py3-jupyter

# Install python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir pandas && \ 
    pip install --no-cache-dir seaborn && \ 
    pip install --no-cache-dir sklearn && \ 
    pip install --no-cache-dir jupyter_contrib_nbextensions

# Install jupyter nbextension and enable used extension
RUN jupyter contrib nbextension install --sys-prefix && \
    jupyter nbextension enable --sys-prefix collapsible_headings/main && \
    jupyter nbextension enable --sys-prefix toc2/main && \
    jupyter nbextension enable --sys-prefix ruler/edit && \
    jupyter nbextension enable --sys-prefix hinterland/hinterland && \
    jupyter nbextension enable --sys-prefix spellchecker/main && \
    jupyter nbextension enable --sys-prefix select_keymap/main
