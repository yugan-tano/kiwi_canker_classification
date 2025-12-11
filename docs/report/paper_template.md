---
title: "基于SE-ResNet的猕猴桃溃疡病图像分类方法"
author: |
  | 张三\textsuperscript{1}, 李四\textsuperscript{1}, 王五\textsuperscript{1}
  | \textsuperscript{1}人工智能学院，XX大学，城市 100000
date: \today
abstract: |
  摘要内容...
keywords: 猕猴桃溃疡病, 图像分类, 深度学习, 注意力机制, SE-ResNet
header-includes:
  - \usepackage{ctex}
  - \usepackage{graphicx}
  - \usepackage{booktabs}
  - \usepackage{multirow}
  - \usepackage{amsmath}
  - \usepackage{algorithm}
  - \usepackage{algorithmic}
  - \usepackage{float}
  - \usepackage{caption}
  - \usepackage{subcaption}
  - \usepackage{geometry}
geometry: "left=3cm,right=2.5cm,top=2.5cm,bottom=2.5cm"
documentclass: article
classoption: 12pt
numbersections: true
toc: true
toc-depth: 3
---

\pagenumbering{Roman}
\begin{abstract}
\input{01-abstract.md}
\end{abstract}

\textbf{关键词：} 猕猴桃溃疡病；图像分类；深度学习；注意力机制；SE-ResNet

\tableofcontents
\newpage
\listoffigures
\listoftables
\newpage
\pagenumbering{arabic}
\setcounter{page}{1}

% ============ 正文开始 ============
\section{引言}
\label{sec:introduction}
\input{02-introduction.md}

\section{相关工作}
\label{sec:related-work}
\input{03-related-work.md}

\section{方法论}
\label{sec:methodology}
\input{04-methodology/01-resnet.md}
\input{04-methodology/02-se-mechanism.md}
\input{04-methodology/03-se-resnet.md}

\section{实验与结果分析}
\label{sec:experiments}
\input{05-experiments/01-dataset.md}
\input{05-experiments/02-setup.md}
\input{05-experiments/03-results.md}

\section{结论与展望}
\label{sec:conclusion}
\input{06-conclusion.md}

% ============ 参考文献 ============
\newpage
\bibliographystyle{IEEEtran}
\bibliography{references}

% ============ 附录 ============
\begin{appendix}
\section{代码实现细节}
\label{app:code}
主要函数实现如下：

\begin{algorithm}[H]
\caption{Squeeze-and-Excitation模块前向传播}
\begin{algorithmic}[1]
\REQUIRE 输入特征图 $X \in \mathbb{R}^{C \times H \times W}$
\ENSURE 重标定后的特征图 $\tilde{X}$
\STATE // Squeeze: 全局平均池化
\STATE $z_c \gets \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_c(i,j)$
\STATE // Excitation: 两层全连接层
\STATE $s \gets \sigma(W_2 \delta(W_1 z))$ \COMMENT{$\delta$为ReLU，$\sigma$为Sigmoid}
\STATE // Scale: 通道注意力加权
\STATE $\tilde{x}_c \gets s_c \cdot x_c$
\STATE \textbf{return} $\tilde{X}$
\end{algorithmic}
\label{alg:se_module}
\end{algorithm}

\section{实验环境配置}
实验所用软硬件环境如表\ref{tab:environment}所示。
\begin{table}[H]
\centering
\caption{实验环境配置}
\label{tab:environment}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{项目} & \textbf{配置} \\
\midrule
操作系统 & Ubuntu 20.04 LTS \\
CPU & Intel Core i7-10700K \\
GPU & NVIDIA RTX 3080 (10GB) \\
内存 & 32GB DDR4 \\
深度学习框架 & PyTorch 1.12.1 \\
Python版本 & 3.9.13 \\
CUDA版本 & 11.6 \\
\bottomrule
\end{tabular}
\end{table}
\end{appendix}
