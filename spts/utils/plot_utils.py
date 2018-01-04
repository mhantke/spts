import h5py
import numpy as np
from matplotlib import pyplot as pypl
import itertools

def plot_cross_distribution(dx, dy, dx0, dy0, rmax, ds, filename_png=None):
    dx_flat = list(itertools.chain.from_iterable(dx))
    dy_flat = list(itertools.chain.from_iterable(dy))
    ranges = [[-rmax-0.5, rmax+0.5], [-0.5, rmax+0.5]]
    bins = [2*rmax/ds+1, rmax/ds+1]
    fig, (ax1, ax2) = pypl.subplots(1, 2, figsize=(15, 3))
    ax1.hist2d(dx_flat, dy_flat, range=ranges, bins=bins, cmap="plasma")
    ax1.set_aspect(1.)
    ax2.scatter(dx0, dy0, color="red")
    ax2.set_xlim(ranges[0][0], ranges[0][1])
    ax2.set_ylim(ranges[1][0], ranges[1][1])
    ax2.set_aspect(1.)
    pypl.show()
    if filename_png is not None:
        fig.savefig(filename_png)    
    
def plot_velocity_distribution(x0_f, y0_f, dx_f, dy_f, x_origin, y_origin, delay_ns, x0, y0, pix_to_um, pix_to_m_per_sec, title=None, filename_png=None):
    sf = 0.7
    fig, (ax2, ax1, ax0) = pypl.subplots(1, 3, figsize=(sf*12, sf*5))#, sharex=True)
    
    assert all(np.isfinite(dx_f)*np.isfinite(dy_f)*np.isfinite(x0_f)*np.isfinite(y0_f))
    v_f = np.sqrt(dx_f**2 + dy_f**2)
    
    v0 = np.median(v_f)
    ax_v0 = pix_to_m_per_sec(v0, delay_ns)
    ax_vmin = ax_v0 - 20
    ax_vmax = ax_v0 + 20
    
    ax_xmin = -300.
    ax_xmax = 300.
    ax_ymin = -500.
    ax_ymax = 500.

    aspect_0 = (ax_ymax-ax_ymin)/(ax_vmax-ax_vmin)
    aspect_1 = (ax_ymax-ax_ymin)/(ax_xmax-ax_xmin)
    aspect_2 = (ax_vmax-ax_vmin)/(ax_xmax-ax_xmin)
    
    ax1.quiver(pix_to_um(x0_f-x_origin), pix_to_um(y0_f-y_origin),
               pix_to_um(dx_f), -pix_to_um(dy_f), 
               pix_to_um(v_f), 
               scale_units = 'xy', scale=1, cmap='autumn_r',
               headwidth=10,
               headlength=10,
              )
    ax1.set_xlim(ax_xmin, ax_xmax)
    ax1.set_ylim(ax_ymax, ax_ymin)
    ax1.set_aspect(1.)    
    ax1.set_ylabel("y [um]")
    ax1.set_xlabel("x [um]")
      
    ax2.scatter(pix_to_um(x0_f-x_origin), pix_to_m_per_sec(v_f, delay_ns), 5, color="gray", marker=".")    
    ax2.set_xlim(ax_xmin, ax_xmax)
    ax2.set_ylim(ax_vmin, ax_vmax)    
    ax2.set_aspect(aspect_1/aspect_2)
    ax2.set_xlabel("x [um]")
    ax2.set_ylabel("Particle velocity [m/s]")
    
    ax0.scatter(pix_to_m_per_sec(v_f, delay_ns), pix_to_um(y0_f-y_origin), 5, color="gray", marker=".")
    ax0.set_ylim(ax_ymax, ax_ymin)
    ax0.set_xlim(ax_vmin, ax_vmax)
    ax0.set_xticks(np.linspace(ax_vmin, ax_vmax, 5))
    ax0.set_aspect(aspect_1/aspect_0)
    ax0.set_ylabel("y [um]")
    ax0.set_xlabel("Particle velocity [m/s]")

    pypl.tight_layout()

    ax0.set_frame_on(False)
    ax1.set_frame_on(False)
    ax2.set_frame_on(False)
    
    #ax0.set_xticks(arange(ax_xmin+100, ax_xmax, 100))
    ax1.set_xticks(np.arange(ax_xmin+100, ax_xmax, 100))
    ax2.set_xticks(np.arange(ax_xmin+100, ax_xmax, 100))
    if title is not None:
        fig.suptitle(title)
    pypl.show()
    if filename_png is not None:
        fig.savefig(filename_png, dpi=400)
    

    
def plot_beam_width(x, y, x_min, x_max, y_min, y_max, y_nbins, x_nbins, ds_name, pix_to_um):
    fig, (ax1, ax2) = pypl.subplots(1, 2, figsize=(4, 3))

    widths_y = np.linspace(y_min, y_max, y_nbins)
    widths = np.zeros_like(widths_y)
    widths_err = np.zeros_like(widths_y)
    window_size = (y_max - y_min) / float(y_nbins-1)
    
    for i,yi in enumerate(widths_y):
        #sel = (y > y_min_bin)*(y <= y_max_bin)
        sel = abs(y-yi) < window_size/2.
        sel *= (y > y_min) * (y < y_max)
        if sel.sum() == 0:
            widths[i] = nan
            widths_err[i] = nan
            continue
        H, xedges = np.histogram(x[sel], range=(x_min, x_max), bins=x_nbins)
        xcenters = xedges[:-1]+(xedges[1]-xedges[0])/2.
        import eval as e
        (A, x0, sigma), H_fit = e.gaussian_fit(xcenters, H)
        widths[i] = 2*np.sqrt(2*np.log(2)) * sigma
        ax1.plot(i*30 + xcenters, i*40 + H)
        ax1.plot(i*30 + xcenters, i*40 + H_fit, color="black")
    
    ax2.plot(pix_to_um(widths_y), pix_to_um(widths))
    #fig.savefig("beam_size_%s.png" % ds_name)
    pypl.show()
