import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import math

fig, ax = plt.subplots(figsize=(5.6,6))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_yticks([])
ax.set_xticks([])

color_green = "#14fb1d"
color_grey = "grey" 
color_red = "#ff0908"
label_green = "Positive result"
label_red = "Negative result"
label_grey = "Neutral or\nno result"
plt.rc('font', family='Helvetica')
marker1 = 'o'
marker2 = 'h'
marker3 = 'H'
size1=100
size2=100
fs = 12

kwargs1 = {'marker': marker1, 's': size1}
kwargs2 = {'marker': marker2, 's': size2}
kwargs3 = {'marker': marker3, 's': size2}

ax.set_xlim([-8,15+5])
ax.set_ylim([-14-2.5,14+2.5])

width = 5
col_gap = 2.5
col1_xoffset = 0.0
col2_xoffset = col1_xoffset + width + col_gap
col3_xoffset = col2_xoffset + width + col_gap
y_offset = 1.0


####### 
# Data from review
#######

num_A = 77
num_B = 60

height_A = math.ceil(num_A / width)
height_B = math.ceil(num_B / width)

x_A = np.arange(num_A) % 5
x_B = np.arange(num_B) % 5

y_A = np.arange(num_A) // 5
y_B = np.arange(num_B) // 5


n_red_A1 = 56
n_grey_A1 = 5
n_green_A1 = 16

n_red_A2 = 10
n_grey_A2 = 8
n_green_A2 = 59

n_red_A3 = 0
n_grey_A3 = 0
n_green_A3 = 77

n_red_B1 = 60
n_grey_B1 = 0
n_red_B2 = 60
n_grey_B2 = 0
n_red_B3 = 7
n_grey_B3 = 53



x_A1green = x_A[0:n_green_A1] + col1_xoffset
y_A1green = y_A[0:n_green_A1] + y_offset
x_A1grey = x_A[n_green_A1:n_green_A1 + n_grey_A1] + col1_xoffset
y_A1grey = y_A[n_green_A1:n_green_A1 + n_grey_A1] + y_offset
x_A1red = x_A[n_green_A1 + n_grey_A1:] + col1_xoffset
y_A1red = y_A[n_green_A1 + n_grey_A1:] + y_offset


x_A2green = x_A[0:n_green_A2] + col2_xoffset
y_A2green = y_A[0:n_green_A2] + y_offset
x_A2grey = x_A[n_green_A2:n_green_A2 + n_grey_A2] + col2_xoffset
y_A2grey = y_A[n_green_A2:n_green_A2 + n_grey_A2] + y_offset
x_A2red = x_A[n_green_A2 + n_grey_A2:] + col2_xoffset
y_A2red = y_A[n_green_A2 + n_grey_A2:] + y_offset


x_A3green = x_A + col3_xoffset
y_A3green = y_A + y_offset


x_B1red = x_B + col1_xoffset
y_B1red = y_B + y_offset
x_B2red = x_B + col2_xoffset
y_B2red = y_B + y_offset

x_B3grey = x_B[0:n_grey_B3] + col3_xoffset
y_B3grey = y_B[0:n_grey_B3] + y_offset
x_B3red = x_B[n_grey_B3:] + col3_xoffset
y_B3red = y_B[n_grey_B3:] + y_offset

#######
# End data
#######




style = "Simple, tail_width=1.5, head_width=12, head_length=8"

kw = dict(arrowstyle=style, color="#747474")
a1 = patches.FancyArrowPatch((col1_xoffset + width/2, y_offset + height_B), (col2_xoffset + width/8, y_offset + height_B),
                             connectionstyle="arc3,rad=-0.75", **kw)

a2 = patches.FancyArrowPatch((col2_xoffset + width/2, y_offset + height_B), (col3_xoffset + width/8, y_offset + height_B),
                             connectionstyle="arc3,rad=-0.75", **kw)

for a in [a1, a2]:
     ax.add_patch(a)

y_bottom = -17

plt.text(-6, -y_offset - height_A/2, "Sample A", fontsize=fs)
plt.text(-6, height_B/2, "Sample B", fontsize=fs)
plt.text(col1_xoffset + width + 0.5, y_offset + height_B + 2.75, "Weak\nbaselines", horizontalalignment='center', fontsize=fs)
plt.text(col2_xoffset + width + 0.5, y_offset + height_B + 2.75, "Outcome\nreporting bias", horizontalalignment='center', fontsize=fs)
plt.text(col1_xoffset - 1.1, y_bottom + 0.7, "(a)", horizontalalignment='center', fontsize=fs)
plt.text(col2_xoffset - 1.1, y_bottom + 0.7, "(b)", horizontalalignment='center', fontsize=fs)
plt.text(col3_xoffset - 1.1, y_bottom + 0.7, "(c)", horizontalalignment='center', fontsize=fs)






ax.scatter(x_A1red, y_bottom+y_A1red, **kwargs1, color=color_red, label = label_red)
ax.scatter(x_A1green, y_bottom+y_A1green, **kwargs2, color=color_green, label = label_green)
ax.scatter(x_A1grey, y_bottom+y_A1grey, **kwargs3, color=color_grey, label = label_grey)
ax.scatter(x_B1red, y_B1red, **kwargs1, color=color_red)

ax.scatter(x_A2red, y_bottom+y_A2red, **kwargs1, color=color_red)
ax.scatter(x_A2grey, y_bottom+y_A2grey, **kwargs3, color=color_grey)
ax.scatter(x_A2green, y_bottom+y_A2green, **kwargs2, color=color_green)
ax.scatter(x_B2red, y_B2red, **kwargs1, color=color_red)

ax.scatter(x_A3green, y_bottom+y_A3green, **kwargs2, color=color_green)
ax.scatter(x_B3red, y_B3red, **kwargs1, color=color_red)
ax.scatter(x_B3grey, y_B3grey, **kwargs3, color=color_grey)

fig.legend(loc = (0.0,0.8), fontsize=fs, frameon=False)
fig.tight_layout()
plt.show()

