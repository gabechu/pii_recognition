#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


# In[2]:


# O: I have lost my card 5199807081286740 and blocked by c4c4:9bac:38a3:886:f173:826c:d16d:e730 can't submit a replacement request. My name is Maisie Griffiths.
# G: I have lost my card xxxxxxxxxxxxxxxx and blocked by xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx can't submit a replacement request. My name is xxxxxxxxxxxxxxxx.
# P: I have lost my card 51998070xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx826c:d16d:e730 can't submit a replacement request. My name is Maisie xxxxxxxxx.


# settings
origin_text = (
    "I have lost my card 5199807081286740 and blocked "
    "by c4c4:9bac:38a3:886:f173:826c:d16d:e730 can't "
    "submit a replacement request. My name is Maisie Griffiths."
)

true_spans = [
    {"text": "5199807081286740",                       "start": 20,  "end": 36,  "type": "credit_card"},
    {"text": "c4c4:9bac:38a3:886:f173:826c:d16d:e730", "start": 52,  "end":90,   "type": "ip_address"},
    {"text": "Maisie Griffiths",                       "start": 138, "end": 154, "type": "person"}
]

pred_spans = [
    {"text": "81286740 and blocked by c4c4:9bac:38a3:886:f173:", "start": 28,  "end": 76,  "type": "ip_address"},
    {"text": "Griffiths",                                        "start": 145, "end": 154, "type": "person"}
]


# # Micro/Global-based
# 
# If we want to evaluate a system's performance in perspective of PII versus non-PII and don't care about labels, this is a method we can adopt. It's very simple.

# In[3]:


# G: I have lost my card xxxxxxxxxxxxxxxx and blocked by xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx can't submit a replacement request. My name is xxxxxxxxxxxxxxxx.
# P: I have lost my card 51998070xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx826c:d16d:e730 can't submit a replacement request. My name is Maisie xxxxxxxxx.

# xxxx --> 1
# everything else --> 0

# G: 0000000000000000000011111111111111110000000000000000111111111111111111111111111111111111110000000000000000000000000000000000000000000000001111111111111111.
# P: 0000000000000000000000000000111111111111111111111111111111111111111111111111000000000000000000000000000000000000000000000000000000000000000000000111111111.


# Calculate character level precision:
# <br>
# $
# \begin{align}
# precision &= \frac{tp}{tp + fp}\\
# &= \frac{len(\text{"81286740"} + \text{"c4c4:9bac:38a3:886:f173:"} + \text{"Griffiths"})}{len(\text{"81286740"} + \text{"c4c4:9bac:38a3:886:f173:"} + \text{"Griffiths"}) + len(\text{" and blocked by "})}
# \end{align}
# $
# 
# Calculate character level recall:
# <br>
# $
# \begin{align}
# recall &= \frac{tp}{tp + fn}\\
# &= \frac{len(\text{"81286740"} + \text{"c4c4:9bac:38a3:886:f173:"} + \text{"Griffiths"})}{len(\text{"81286740"} + \text{"c4c4:9bac:38a3:886:f173:"} + \text{"Griffiths"}) + len(\text{"51998070"} + \text{"826c:d16d:e730"} + \text{"Maisie "})}
# \end{align}
# $

# In[4]:


def build_binary(text_length, spans):
    """Binary represenatation for text. 
    Spans are labeled with 1 and non-spans are label with 0."""
    array = [0] * text_length

    for span in spans:
        s = span["start"]
        e = span["end"]
        array[s:e] = [1] * (e - s)

    return array


# In[5]:


def calculate_micro_scores(text, true_spans, pred_spans):
    text_length = len(text)
    true_binary = build_binary(text_length, true_spans)
    pred_binary = build_binary(text_length, pred_spans)
    
    precision = precision_score(true_binary, pred_binary)
    recall = recall_score(true_binary, pred_binary)
    f1 = f1_score(true_binary, pred_binary)
    
    return precision, recall, f1

precision, recall, f1 = calculate_micro_scores(origin_text, true_spans, pred_spans)
f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}"


# ## Problem with the micro method

# In[6]:


text_p = "My name is Gabe and this is my email very_long_email_gabechu@gmail.com."

true_spans_p = [
    {"text": "Gabe",              "start": 11,   "end": 15,  "type": "person"},
    {"text": "gabechu@gmail.com", "start": 37,   "end":70,   "type": "email"},
]

pred_spans_p = [
    {"text": "gabechu@gmail.com", "start": 37,  "end":70,   "type": "email"},
]


# In[7]:


precision, recall, f1 = calculate_micro_scores(text_p, true_spans_p, pred_spans_p)
f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}"


# # Macro/Entity-based
# To fix the underestimating and overestimating issues of the micro method, we propose a macro approach that iteratively calculates scores for each entity.
# 
# ## Boundary detection

# In[8]:


# O: I have lost my card 5199807081286740 and blocked by c4c4:9bac:38a3:886:f173:826c:d16d:e730 can't submit a replacement request. My name is Maisie Griffiths.

# Entity: 81286740 and blocked by c4c4:9bac:38a3:886:f173:
# G: I have lost my card xxxxxxxxxxxxxxxx and blocked by xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx can't submit a replacement request. My name is xxxxxxxxxxxxxxxx.
# P: I have lost my card 51998070xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx826c:d16d:e730 can't submit a replacement request. My name is Maisie Griffiths.

# Entity: Griffiths
# G: I have lost my card xxxxxxxxxxxxxxxx and blocked by xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx can't submit a replacement request. My name is xxxxxxxxxxxxxxxx.
# P: I have lost my card 5199807081286740 and blocked by c4c4:9bac:38a3:886:f173:826c:d16d:e730 can't submit a replacement request. My name is Maisie xxxxxxxxx.


# In[9]:


def calculate_ave_boundary_precisions(text, true_spans, pred_spans):
    text_length = len(text)
    true_binary = build_binary(text_length, true_spans)
    
    boundary_precisions = []
    for pred_span in pred_spans:
        pred_span_binary = build_binary(text_length, [pred_span])
        precision = precision_score(true_binary, pred_span_binary)
        boundary_precisions.append(precision)
    return boundary_precisions

boundary_precisions = calculate_ave_boundary_precisions(origin_text, true_spans, pred_spans)
ave_boundary_precision = sum(boundary_precisions) / len(boundary_precisions)


# In[10]:


# O: I have lost my card 5199807081286740 and blocked by c4c4:9bac:38a3:886:f173:826c:d16d:e730 can't submit a replacement request. My name is Maisie Griffiths.

# Entity: 5199807081286740
# G: I have lost my card xxxxxxxxxxxxxxxx and blocked by c4c4:9bac:38a3:886:f173:826c:d16d:e730 can't submit a replacement request. My name is Maisie Griffiths.
# P: I have lost my card 51998070xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx826c:d16d:e730 can't submit a replacement request. My name is Maisie xxxxxxxxx.

# Entity: c4c4:9bac:38a3:886:f173:826c:d16d:e730
# G: I have lost my card 5199807081286740 and blocked by xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx can't submit a replacement request. My name is Maisie Griffiths.
# P: I have lost my card 51998070xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx826c:d16d:e730 can't submit a replacement request. My name is Maisie xxxxxxxxx.

# Entity: Maisie Griffiths
# G: I have lost my card 5199807081286740 and blocked by c4c4:9bac:38a3:886:f173:826c:d16d:e730 can't submit a replacement request. My name is xxxxxxxxxxxxxxxx.
# P: I have lost my card 51998070xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx826c:d16d:e730 can't submit a replacement request. My name is Maisie xxxxxxxxx.


# In[11]:


def calculate_ave_boundary_recall(text, true_spans, pred_spans):
    text_length = len(text)
    pred_binary = build_binary(text_length, pred_spans)

    boundary_recalls = []
    for true_span in true_spans:
        true_span_binary = build_binary(text_length, [true_span])
        recall = recall_score(true_span_binary, pred_binary)
        boundary_recalls.append(recall)
    return boundary_recalls

boundary_recalls = calculate_ave_boundary_recall(origin_text, true_spans, pred_spans)
ave_boundary_recall = sum(boundary_recalls) / len(boundary_recalls)


# In[12]:


boundary_f1 = (2 * ave_boundary_precision * ave_boundary_recall) / (ave_boundary_precision + ave_boundary_recall)

f"For boundary detection: ave-precision: {ave_boundary_precision:.4f}, ave-recall: {ave_boundary_recall:.4f}, f1: {boundary_f1:.4f}"


# In[13]:


# For the problematic example
boundary_precisions = calculate_ave_boundary_precisions(text_p, true_spans_p, pred_spans_p)
ave_boundary_precision = sum(boundary_precisions) / len(boundary_precisions)

boundary_recalls = calculate_ave_boundary_recall(text_p, true_spans_p, pred_spans_p)
ave_boundary_recall = sum(boundary_recalls) / len(boundary_recalls)

boundary_f1 = (2 * ave_boundary_precision * ave_boundary_recall) / (ave_boundary_precision + ave_boundary_recall)
f"For boundary detection: ave-precision: {ave_boundary_precision:.4f}, ave-recall: {ave_boundary_recall:.4f}, f1: {boundary_f1:.4f}"


# ## Type identification
# 
# What if we want to know the performance of LOCATION for example.

# In[14]:


# Instead of implementing this in binary, we can code it up in multi-class

# 1 --> type credit_card
# 2 --> type ip_address
# 3 --> type person

# O: I have lost my card 5199807081286740 and blocked by c4c4:9bac:38a3:886:f173:826c:d16d:e730 can't submit a replacement request. My name is Maisie Griffiths.
# G: I have lost my card 1111111111111111 and blocked by 22222222222222222222222222222222222222 can't submit a replacement request. My name is 3333333333333333.
# P: I have lost my card 51998070222222222222222222222222222222222222222222222222826c:d16d:e730 can't submit a replacement request. My name is Maisie 333333333.


# In[15]:


mapping = {
    "credit_card": 1,
    "ip_address": 2,
    "person": 3,
}

def build_multinomial(text_length, spans, mapping):
    array = [0] * text_length
    
    for span in spans:
        s = span["start"]
        e = span["end"]
        class_name = mapping[span["type"]]
        array[s : e] = [class_name] * (e - s)
    
    return array


# In[16]:


def calculate_type_precisions(text, true_spans, pred_spans):
    text_length = len(text)
    true_multi = build_multinomial(text_length, true_spans, mapping)
    
    type_precisions = []
    for pred_span in pred_spans:
        pred_span_multi = build_multinomial(text_length, [pred_span], mapping)
        span_class = mapping[pred_span["type"]]
        # index 0 because the list contains only 1 element
        precision = precision_score(true_multi, pred_span_multi, average=None, labels=[span_class])[0]
        type_precisions.append(precision)
        
    return type_precisions

type_precisions = calculate_type_precisions(origin_text, true_spans, pred_spans)
ave_type_precision = sum(type_precisions) / len(type_precisions)


# In[17]:


def calculate_type_recalls(text, true_spans, pred_spans):
    text_length = len(text)
    pred_multi = build_multinomial(text_length, pred_spans, mapping)
    
    type_recalls = []
    for true_span in true_spans:
        true_span_multi = build_multinomial(text_length, [true_span], mapping)
        span_class = mapping[true_span["type"]]
        # index 0 because the list contains only 1 element
        recall = recall_score(true_span_multi, pred_multi, average=None, labels=[span_class])[0]
        type_recalls.append(recall)
    return type_recalls

type_recalls = calculate_type_recalls(origin_text, true_spans, pred_spans)
ave_type_recall = sum(type_recalls) / len(type_recalls)


# In[18]:


type_f1 = (2 * ave_type_precision * ave_type_recall) / (ave_type_precision + ave_type_recall)

f"For type identification: ave-precision: {ave_type_precision:.4f}, ave-recall: {ave_type_recall:.4f}, f1: {type_f1:.4f}"


# ## Performance of boundary detection and type identification for every entity

# In[19]:


# add precison scores
for i, span in enumerate(pred_spans):
    span.update({"precision": type_precisions[i]})
    
pred_spans


# In[20]:


# add recall scores
for i, span in enumerate(true_spans):
    span.update({"recall": type_recalls[i]})
    
true_spans


# In[21]:


# f1 on ip_address
f1_ip_address = (2 * 0.3333 * 0.3988) / (0.3333 + 0.3988)
f1_ip_address


# In[ ]:




