import os
import re

import time
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import *

import nibabel as nb
from dipy.io.image import load_nifti

import cc3d

import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfile, askdirectory
from tqdm import tqdm

import warnings

warnings.simplefilter("ignore")

# data, _, img = load_nifti(CT_path, return_img=True)

# def save_to_file(new_data, suffix=''):
#     save_img = nb.Nifti1Image(new_data, img.affine)
#     t = str(time.asctime()).replace(' ', '_').replace(':', '_')
#     if suffix: suffix += '_'
#     nb.save(save_img, f"Transformed_Image_{suffix}{t}.nii")

# th = 200
# def plot_change(orig, processed):
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     ax1.imshow(orig[:, -1:0:-1, 101].T), ax1.set_title('Original')
#     ax2.imshow(processed[:, -1:0:-1, 101].T), ax2.set_title(f'Above threshold: {th}')
''
##################################################################
PIXEL_VALUE_THRESHOLD = 200  # for binary mask on original image
MAX_REGION_SIZE = 200  # for region growing algorithm

ELECTRODES = {'LAH': [24, 30, 31, 32, 36, 37, 38, 39], 'LTP': [76, 82, 85, 90, 93, 97],
              'RA': [25, 26, 33, 40, 42, 43, 45, 49], 'RAH': [16, 18, 19, 22, 23, 29, 35, 20],
              'RP1': [102, 99, 96, 94, 91, 86, 80, 78], 'RP2': [81, 87, 92, 98, 101, 105, 107, 110],
              'RP3': [75, 77, 83, 89, 95, 100, 103], 'RPO': [56, 58, 59, 60, 61, 62, 63, 64],
              'RTO': [41, 44, 46, 47, 48, 50, 51, 52], 'RTP': [65, 66, 68, 69, 70, 71, 72, 73]}


# ELECTRODES = {} #TODO: uncomment

##################################################################
class NiftiLoader:
    def __init__(self, path, th=200):
        self.path = path
        self.data, self.affine = load_nifti(self.path)
        self.th, self.mask, self.labels = th, None, None
        if 0 < th < 256: self.th = int(th)
        self.mask = (self.data > self.th / 255 * self.data.max()).astype('int32')
        self.labels = cc3d.connected_components(self.mask)


##################################################################
def region_grow(vol, start_point, mask=None, epsilon=60, fill_with=1, max_reg_size=MAX_REGION_SIZE):
    """
        `vol` your already segmented 3d-lungs, using one of the other scripts
        `mask` you can start with all 1s, and after this operation, it'll have 0's where you need to delete
        `start_point` a tuple of ints with (z, y, x) coordinates
        `epsilon` the maximum delta of conductivity between two voxels for selection - 80 is a good tradeoff between speed and accuracy
        `fill_with` value to set in `mask` for the appropriate location in vol that needs to be flood filled
    """
    if mask is None: mask = np.zeros(vol.shape)
    sizex, sizey, sizez = vol.shape[0] - 1, vol.shape[1] - 1, vol.shape[2] - 1
    cvoxel = vol[start_point[0], start_point[1], start_point[2]]
    items, visited = [], []

    def enqueue(item):
        if item not in visited: items.insert(0, item)

    def dequeue():
        s = items.pop()
        visited.append(s)
        return s

    check = lambda v1: abs(cvoxel - v1) * 255 / (vol.max() - vol.min()) < epsilon

    enqueue((start_point[0], start_point[1], start_point[2]))

    while items:
        x, y, z = dequeue()
        # voxel = vol[x, y, z]
        mask[x, y, z] = fill_with

        if x < sizex:
            if check(vol[x + 1, y, z]):  enqueue((x + 1, y, z))

        if x > 0:
            if check(vol[x - 1, y, z]):  enqueue((x - 1, y, z))

        if y < sizey:
            if check(vol[x, y + 1, z]):  enqueue((x, y + 1, z))

        if y > 0:
            if check(vol[x, y - 1, z]):  enqueue((x, y - 1, z))

        if z < sizez:
            if check(vol[x, y, z + 1]):  enqueue((x, y, z + 1))

        if z > 0:
            if check(vol[x, y, z - 1]):  enqueue((x, y, z - 1))

        if len(visited) > max_reg_size:
            print(f'Region size exceeds maximum size of {max_reg_size} voxels')
            return visited, mask
    return visited, mask


##################################################################
class Probes:
    def __init__(self, labels=None, loader=None, patient_number=0, electrodes=None):
        if electrodes is None: electrodes = ELECTRODES

        # this inits the electrode related stuff
        self.labels_img = labels
        self.electrodes = electrodes
        self.current_probe = None
        self.subject_number = patient_number
        # this is for the XYZ coords save
        self.channel_numbers_by_name = {}
        self.vhdr_path = ''
        # self.vhdr_path = 'N:\data\Intracranial\PatientsData\BIDSFormattedProjects\DominoProject\sub-034\ses-01\ieeg-Macro\sub-034_task-Domino_ses-01_ieeg-Macro.vhdr'

        # this inits the image related stuff
        self.loader = loader
        self.affine = loader.affine

        self.centers = None
        if self.labels_img is not None:
            self.center_voxel = ([(x - 1) // 2 for x in self.labels_img.shape])
            self.init_centers()

    def __repr__(self):
        return f"{[*self.electrodes.keys()]}"

    def init_centers(self):
        self.centers = {
            name: [np.argwhere(self.labels_img == label).mean(axis=0).astype(int).tolist() for label in labels]
            for name, labels in self.electrodes.items()}

    def get_probe_centers(self, n, cent, probe_image=None):
        new_centers = []
        for c in tqdm(cent, desc=f'{n}\'s electrodes'):
            region, probe_image = region_grow(self.loader.data, start_point=c, mask=probe_image, epsilon=60)
            new_centers.append(np.mean(region, axis=0).astype('int').tolist())
        self.centers[n] = new_centers

        return probe_image

    def _probe_to_nifti(self, n, cent=None):
        probe_image = np.zeros(self.labels_img.shape)
        probe_image = self.get_probe_centers(n, cent, probe_image)
        save_img = nb.Nifti1Image(probe_image, self.affine)
        if 'probes' not in os.listdir(os.getcwd()): os.mkdir('probes')
        nb.save(save_img, f"probes/{n}.nii")

    def save_all_to_nifti(self, event):
        if not self.electrodes: raise ValueError('No Electrodes')
        print('[Saving probes - please wait]')

        # self.centers = self.init_centers()
        for n, cent in tqdm(self.centers.items(), desc=f'probes'):
            self._probe_to_nifti(n, cent)
            print(f'[Saved {n} to nifti]\n')

    def save_probe_to_nifti(self, event):
        self._probe_to_nifti(self.current_probe, self.centers[self.current_probe])

    def order_probe(self, n, cent):
        self.get_probe_centers(n, cent)

        compare = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))
        distances = [compare(c, self.center_voxel) for c in cent]

        sorted_indices = np.argsort(distances).tolist()
        new_centers = [cent[i] for i in sorted_indices]
        self.centers[n] = new_centers
        # print(self.centers[n])

    def order_centers(self):
        for n, cent in self.centers.items():
            self.order_probe(n, cent)

    def save_xyz_by_name(self, event):
        # global topdir
        os.chdir(topdir + '/anat')
        dir_name = 'probes'
        if dir_name not in os.listdir(os.getcwd()):
            os.mkdir(dir_name)

        save_file_name = f'sub-{self.subject_number}_ses-01_Macro_elec_XYZcoord'
        fmt_sign = ','

        fmt = input('choose file format: (default= \'csv\') \n0) csv\n1) tsv\n')
        print(fmt)
        if fmt not in ['0', '1']: fmt = 0
        fmt = int(fmt)
        if fmt: fmt_sign = '\t'; save_file_name += '.tsv'
        else: save_file_name += '.csv'

        if save_file_name in os.listdir(topdir + '/anat/' + dir_name):
            print('[File name exists - stopped saving]')
            return

        if not self.vhdr_path: self.vhdr_path = askopenfilename(
            title=f'Open \'Macro.vhdr\' file for sub-{self.subject_number}', initialdir=topdir + '/ses-01/ieeg-Macro')

        if not self.channel_numbers_by_name:
            _, self.channel_numbers_by_name = self.ch_names_from_vhdr(self.vhdr_path)

        reorder = input('do you want to reorder the centers? (default=0) \n0) No\n1) Yes\n')
        if reorder not in ['0', '1']: reorder = 0
        reorder = int(reorder)
        if reorder: self.order_centers()

        with open(dir_name + '/' + save_file_name, 'w') as f:
            # header: Patient Name:
            f.write(f'Subject:{fmt_sign}sub-{self.subject_number}{fmt_sign}{fmt_sign}\n')
            f.write(f'Channel_num{fmt_sign}Channel_name{fmt_sign}X{fmt_sign}Y{fmt_sign}Z\n')

            for k, vs in self.centers.items():
                for i, v in enumerate(vs):
                    chan_name = f'{k}{i + 1}'
                    if any(char.isdigit() for char in k): chan_name = f'{k}_{i + 1}'
                    try:
                        f.write(
                            f'{self.channel_numbers_by_name[chan_name]}{fmt_sign}{chan_name}{fmt_sign}{v[0]}{fmt_sign}{v[1]}{fmt_sign}{v[2]}\n')
                    except Exception as e:
                        f.write(f',{chan_name}{fmt_sign}{v[0]}{fmt_sign}{v[1]}{fmt_sign}{v[2]}\n')
                        print(e)
            print(f'[File: {save_file_name} saved successfully]')

    def ch_names_from_vhdr(self, path, verbose=False):
        with open(path) as f:
            lines = [l.strip() for l in f if l.startswith('Ch')]

        ch_names, ch_nums = [], {}
        channel_regex_macro = re.compile('^Ch(\d+)=EEG_([a-zA-Z]+)(\d)?_?(?:post)?(\d)?,,$')
        for l in lines:
            m = re.match(channel_regex_macro, l)
            if m:
                chan_number = m.group(1)
                new_name = m.group(2)
                name_number = m.group(3)
                ch_num_name = new_name + name_number

                if m.group(4):
                    new_name += name_number
                    ch_num_name = new_name + '_' + m.group(4)

                if new_name not in ch_names:
                    ch_names.append(new_name)

                ch_nums[ch_num_name] = chan_number

            elif verbose:
                print(f'{l} is not a valid channel name (Ch<Channel number>=EEG_<Name>_<Reference channel name>,,)')
        ch_names = sorted(ch_names)

        return ch_names, ch_nums


##################################################################
class ProbeMarker:

    def __init__(self, path):
        # load input data
        self.path = path.replace('\\', '/')
        _, self.ax = plt.subplots(2, 1)
        self.loader = NiftiLoader(path, th=PIXEL_VALUE_THRESHOLD)
        self.orig = self.loader.data
        self.labels = self.loader.labels
        self.patient_number = self._get_patient_number()

        # this initializes the probes
        self.probes = Probes(self.labels, self.loader, self.patient_number)

        # define the index and 2D image slices
        _, _, self.slices, = self.orig.shape
        self.ind = self.slices // 2
        self.ct_img = self.ax[0].imshow(self.orig[:, :, self.ind, ])
        self.labels_img = self.ax[1].imshow(self.labels[:, :, self.ind, ])
        self.ax[0].set_title(f'Slice number {self.ind + 1} of {self.slices}')
        plot_center = [self.probes.center_voxel[1], self.probes.center_voxel[0], self.probes.center_voxel[2]]
        self.ax[0].scatter(*plot_center, marker='*', edgecolors='w', linewidths=0.5)
        self.ax[1].scatter(*plot_center, marker='*', edgecolors='w', linewidths=0.5)

        # this sets the widgets and their functionality
        self.inax = False
        self._set_widgets()
        self._set_functionality()

        # this sets the presentable data tkinter widget
        self.tk_master = None
        self.current_labels = []
        self._set_text()

        self.probes.vhdr_path = ''

    def __repr__(self):
        return f'iEEG probe selection tool, file: {self.path}'

    def _set_widgets(self):
        l_most_left = 0.045
        r_most_left = 0.75

        # 2D slice navigation
        self.buttons = {'Up': Button(plt.axes([l_most_left, 0.825, 0.08, 0.05]), 'Up'),
                        'Down': Button(plt.axes([l_most_left, 0.745, 0.08, 0.05]), 'Down')}
        self.slider = Slider(plt.axes([l_most_left + 0.12, 0.525, 0.05, 0.35]), 'Slice',
                             valmin=1, valmax=self.slices, valinit=self.ind + 1,
                             valstep=1, orientation='vertical')
        self.SliceTextBox = TextBox(plt.axes([l_most_left, 0.605, 0.08, 0.05]), '')
        self.SliceTextBoxLabel = plt.text(0, 1.2, 'Go To:')
        self.buttons['Go'] = Button(plt.axes([l_most_left, 0.525, 0.08, 0.05]), 'Go')

        # Probe handeling
        self.ProbeNameTextBox = TextBox(plt.axes([l_most_left, 0.385, 0.17, 0.05]), label='')
        self.SliceTextBoxLabel = plt.text(0, 1.2, 'Probe Name:')
        self.buttons['Add probe'] = Button(plt.axes([l_most_left, 0.315, 0.17, 0.05]), 'Add Probe')
        self.buttons['Rename probe'] = Button(plt.axes([l_most_left, 0.245, 0.17, 0.05]), 'Rename Probe')
        self.buttons['Remove probe'] = Button(plt.axes([l_most_left, 0.175, 0.17, 0.05]), 'Remove Probe')
        self.buttons['Load probes'] = Button(plt.axes([l_most_left, 0.105, 0.17, 0.05]), 'Load Probes')

        # label selection
        self.buttons['Show probes'] = Button(plt.axes([r_most_left, 0.53, 0.22, 0.05]), 'Show Probes')
        self.selector = RectangleSelector(self.ax[1], self.box_selection_callback,
                                          drawtype='box', useblit=True,
                                          button=[1],  # disable middle button
                                          minspanx=5, minspany=5,
                                          spancoords='pixels',
                                          interactive=True)

        # label handling
        self.buttons['Add electrodes'] = Button(plt.axes([r_most_left, 0.41, 0.22, 0.05]), 'Add Electrodes')
        self.buttons['Remove electrodes'] = Button(plt.axes([r_most_left, 0.34, 0.22, 0.05]), 'Remove Electrodes')

        # saving probe centers
        self.buttons['Save all to Nifti'] = Button(plt.axes([r_most_left, 0.27, 0.22, 0.05]), 'Save all to Nifty')
        self.buttons['Save probe to Nifti'] = Button(plt.axes([r_most_left, 0.20, 0.22, 0.05]), 'Save probe to Nifty')
        self.buttons['Save coordinates'] = Button(plt.axes([r_most_left, 0.13, 0.22, 0.05]), 'Save all coordinates')
        # loading probe names from a file

    def _set_functionality(self):
        fig = self.ax[0].axes.figure

        # these are the different scrolling options
        self.buttons['Down'].on_clicked(self.button_down)
        self.buttons['Up'].on_clicked(self.button_up)
        self.buttons['Go'].on_clicked(self.goto_slice)
        self.SliceTextBox.on_submit(self.goto_slice)
        self.slider.on_changed(self.slider_index)
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        # this takes care of probe addition and removal
        self.buttons['Add probe'].on_clicked(self.probe_adder)
        # self.ProbeNameTextBox.on_submit(self.probe_adder)
        # self.ProbeNameTextBox.on_text_change(self.probe_name_change_update)
        self.buttons['Rename probe'].on_clicked(self.probe_rename)
        self.buttons['Remove probe'].on_clicked(self.probe_remover)
        self.buttons['Show probes'].on_clicked(self.probe_show_tk)
        self.buttons['Load probes'].on_clicked(self.probe_load)

        # this takes care of electrode selection
        # fig.canvas.mpl_connect('button_press_event', self.on_Lclick_in_label_img)

        # this takes care of electrode addition and removal
        self.buttons['Add electrodes'].on_clicked(self.electrode_adder)
        self.buttons['Remove electrodes'].on_clicked(self.electrode_remover)

        # this takes care of saving the probes
        self.buttons['Save all to Nifti'].on_clicked(self.probes.save_all_to_nifti)
        self.buttons['Save probe to Nifti'].on_clicked(self.probes.save_probe_to_nifti)
        self.buttons['Save coordinates'].on_clicked(self.probes.save_xyz_by_name)

    def _set_text(self):
        h = 6.7 + 1.5 + 1.5
        self.text = {
            'Probe title': plt.text(0, 5 + h, "Selected Probe:"),
            'Probe var': plt.text(0, 4 + h, "--Choose Probe Name--"),
            'Label title': plt.text(0, 3 + h, "Registered Labels:"),
            'Label var': plt.text(0, 2 + h, "[]"),
            'Selection title': plt.text(0, 1 + h, "Selected Labels:"),
            'Selection var': plt.text(0, 0 + h, "[]"),
        }
        # self.textbox_text = ''

    def on_scroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def button_up(self, event):
        # func for up button press
        self.ind = (self.ind + 1) % self.slices
        self.update()

    def button_down(self, event):
        # func for down button press
        self.ind = (self.ind - 1) % self.slices
        self.update()

    def slider_index(self, event):
        # func for slider index change
        self.ind = int(self.slider.val) - 1
        self.update()

    def goto_slice(self, event):
        # print(event)
        num = self.SliceTextBox.text
        if num.isnumeric():
            num = int(num)
        else:
            return
        if num < 0: num = 0
        if num > self.slices: num = self.slices - 1
        self.ind = num - 1
        self.update()
        self.SliceTextBox.set_val('')

    def probe_adder(self, event):
        name = self.ProbeNameTextBox.text
        if name.lower() not in [n.lower() for n in self.probes.electrodes.keys()] and name.strip() != '' \
                and name != "--Choose Probe Name--":
            self.probes.current_probe = name
            self.probes.electrodes[self.probes.current_probe] = []
            self.ProbeNameTextBox.set_val('')
            self.text['Probe var'].set_text(name)
            self.text['Label var'].set_text(self.probes.electrodes[name])
        else:
            print('probe name invalid')

    def probe_rename(self, event):
        # plt.show()
        # new = self.textbox_text
        new = self.ProbeNameTextBox.text
        if new.lower() in [n.lower() for n in self.probes.electrodes.keys()] and new.strip() != '':
            print(f'{new} is already in the electrode list')
            self.ProbeNameTextBox.set_val('')
            return
        curr = self.probes.current_probe

        if new and curr != '--Choose Probe Name--' and curr is not None:
            self.probes.electrodes[new] = self.probes.electrodes.pop(curr)
            self.probes.centers[new] = self.probes.centers.pop(curr)

            self.probes.current_probe = new
            self.text['Probe var'].set_text(str(new))
            self.ProbeNameTextBox.set_val('')

    def probe_remover(self, event):
        if self.probes.current_probe in self.probes.electrodes.keys():
            del self.probes.electrodes[self.probes.current_probe]
            self.probes.current_probe = None
            self.text['Probe var'].set_text("--Choose Probe Name--")
            self.text['Label var'].set_text("[]")
            # if self.tk_master is not None: self.tk_master.destroy()

    def probe_show_tk(self, event):
        self.tk_master = tk.Tk()
        # self.tk_master.title("Probes")
        # self.tk_master.geometry('200x30')

        drop_menu_var = tk.StringVar(self.tk_master)
        drop_menu_var.set("--Choose Probe Name--")
        probe_menu = tk.OptionMenu(self.tk_master, drop_menu_var, "--Choose Probe Name--",
                                   *self.probes.electrodes.keys())
        probe_menu.pack()

        # this let's you toggle the auto close functionality (of the probe show menu)
        chk_var = tk.BooleanVar(self.tk_master)
        auto_close_cb = tk.Checkbutton(self.tk_master, text='Close on choice', variable=chk_var, )
        auto_close_cb.select()
        auto_close_cb.pack()

        def on_option_selection(*args):
            self.probes.current_probe = drop_menu_var.get()
            self.text['Probe var'].set_text(str(self.probes.current_probe))
            if self.probes.current_probe != '--Choose Probe Name--':
                self.text['Label var'].set_text(str(self.probes.electrodes[self.probes.current_probe]))
            else:
                self.text['Label var'].set_text("[]")
            plt.draw()
            if chk_var.get(): self.tk_master.destroy()

        drop_menu_var.trace('w', on_option_selection)
        self.tk_master.mainloop()

    def probe_load(self, event, verbose=False):
        tk.Tk().withdraw()
        self.probes.vhdr_path = askopenfilename(title='Open \'Macro.vhdr\' file',
                                                initialdir=os.getcwd() + '/ses-01/ieeg-Macro')
        print(f'[Loading probes from: {self.probes.vhdr_path}]')

        names, self.probes.channel_numbers_by_name = self.probes.ch_names_from_vhdr(self.probes.vhdr_path, verbose)

        self.probes.electrodes = {name: [] for name in names}
        print('[Probe names successfully loaded]')

    def electrode_adder(self, event):
        if self.probes.current_probe and self.current_labels:
            for l in self.current_labels:
                if l not in self.probes.electrodes[self.probes.current_probe]:
                    self.probes.electrodes[self.probes.current_probe].append(l)
            self.text['Label var'].set_text(str(self.probes.electrodes[self.probes.current_probe]))
        self.update()

    def electrode_remover(self, event):
        if self.probes.current_probe and self.current_labels:
            for l in self.current_labels:
                if l in self.probes.electrodes[self.probes.current_probe]:
                    self.probes.electrodes[self.probes.current_probe].remove(l)
            self.text['Label var'].set_text(str(self.probes.electrodes[self.probes.current_probe]))
        self.update()

    def on_Lclick_in_label_img(self, event):
        if not event.xdata or not event.ydata: return
        y, x = int(event.xdata), int(event.ydata)
        if y == 0: return

        self.current_labels = [self.labels[x, y, self.ind]]
        # print(self.current_labels)
        if 0 in self.current_labels: self.current_labels.remove(0)
        if self.current_labels: self.text['Selection var'].set_text(str(self.current_labels))

    def box_selection_callback(self, eclick, erelease):
        y1, x1 = int(eclick.xdata), int(eclick.ydata)
        y2, x2 = int(erelease.xdata), int(erelease.ydata)
        # print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        # print(f" The buttons you used were: {eclick.button} {erelease.button}")

        self.current_labels = np.unique(self.labels[x1:x2, y1:y2, self.ind]).tolist()
        if 0 in self.current_labels: self.current_labels.remove(0)
        if self.current_labels: self.text['Selection var'].set_text(str(self.current_labels))

    def update(self):
        # set slider value to current index value
        if self.slider.val != self.ind + 1: self.slider.set_val(self.ind + 1)

        # plot the images corresponding the current index
        self.ct_img.set_data(self.orig[:, :, self.ind, ])
        self.ct_img.axes.figure.canvas.draw()
        # self.ct_img.scatter()

        self.labels_img.set_data(self.labels[:, :, self.ind, ])
        self.labels_img.axes.figure.canvas.draw()

        self.ax[0].set_title(f'Slice number {self.ind + 1} of {self.slices}')

    def _get_patient_number(self):
        re_d = re.compile('.*/D(\d{3})/*')
        d = re.match(re_d, self.path)
        if d: return d.group(1)

        re_sub = re.compile('.*/sub-(\d{3})/*')
        s = re.match(re_sub, self.path)
        if s: return s.group(1)

        return input('Choose patient number:')


if __name__ == '__main__':
    tk.Tk().withdraw()
    global topdir
    topdir = askdirectory(title='Select Subject Directory')
    os.chdir(topdir)
    CT_path = askopenfilename(title='Open \'.nii\' (anatomy) file', initialdir=os.getcwd() + '/anat/')
    # CT_path = 'C:/Users/idanl/Desktop/iEEG_Probe_Marker_Tool/D034/rCT.nii'

    print(f'[Opening file @ {CT_path}]')
    marker = ProbeMarker(CT_path)
    plt.show()

    # AAA = marker.probes
    # AAA.save_xyz_by_name(1)
