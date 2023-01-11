#!/usr/bin/env /usr/local/sci/bin/python2.7
'''
this one plots u-aw447 UKESM1 4xCO2 run
compared with HadGEM2-ES cg772

CDJ. April 2018

Updated Oct 2018 to use new run bb446
'''

#Load all modules required
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
#import iris
#import iris.plot as iplt
#import iris.quickplot as qplt
#import iris.coord_categorisation
#import iris.analysis.cartography
#import cartopy.crs as ccrs
#%matplotlib inline
from scipy import stats
import iris.analysis.stats
import scipy.stats


# UKESM
#dir='/home/h01/cburton/IBACKUP/MECA6/'
dir='/scratch/cburton/scratch/UKESM/CSSP/'


run='u-aw310_1'
y,netcl_aw310,netcf_aw310,osw_aw310,closw_aw310,swcf_aw310,olw_aw310,clolw_aw310,lwcf_aw310,net_aw310 = \
                       np.loadtxt(dir+run+'/'+'toa.dat',skiprows=1, max_rows=100).T
y,tg_aw310,tl_aw310,to_aw310,pg_aw310,pl_aw310,po_aw310 = \
                       np.loadtxt(dir+run+'/'+'t_ppn.dat',skiprows=1, max_rows=100).T
run='u-ca082_1'
y,netcl_ca082,netcf_ca082,osw_ca082,closw_ca082,swcf_ca082,olw_ca082,clolw_ca082,lwcf_ca082,net_ca082 = \
                       np.loadtxt(dir+run+'/'+'toa.dat',skiprows=1, max_rows=100).T
y,tg_ca082,tl_ca082,to_ca082,pg_ca082,pl_ca082,po_ca082 = \
                       np.loadtxt(dir+run+'/'+'t_ppn.dat',skiprows=1, max_rows=100).T
run='u-bb446_1'
y,netcl_bb446,netcf_bb446,osw_bb446,closw_bb446,swcf_bb446,olw_bb446,clolw_bb446,lwcf_bb446,net_bb446 = \
                       np.loadtxt(dir+run+'/'+'toa.dat',skiprows=1, max_rows=100).T
y,tg_bb446,tl_bb446,to_bb446,pg_bb446,pl_bb446,po_bb446 = \
                       np.loadtxt(dir+run+'/'+'t_ppn.dat',skiprows=1, max_rows=100).T
run='u-cg772_1'
y,netcl_cg772,netcf_cg772,osw_cg772,closw_cg772,swcf_cg772,olw_cg772,clolw_cg772,lwcf_cg772,net_cg772 = \
                       np.loadtxt(dir+run+'/'+'toa.dat',skiprows=1, max_rows=100).T
y,tg_cg772,tl_cg772,to_cg772,pg_cg772,pl_cg772,po_cg772 = \
                       np.loadtxt(dir+run+'/'+'t_ppn.dat',skiprows=1, max_rows=100).T

#4XCO2 - PiC
#ctr_t_pts = (tg_bb446 - np.mean(tg_aw310))/2
#ctr_f_pts = net_bb446 - np.mean(net_aw310)
#fire_t_pts = (tg_cg772 - np.mean(tg_ca082))/2
#fire_f_pts = net_cg772 - np.mean(net_ca082)

#4XCO2 - PiC # CHECK 1
ctr_t_pts = (tg_bb446 - tg_aw310)/2
ctr_f_pts = net_bb446 - net_aw310
fire_t_pts = (tg_cg772 - tg_ca082)/2
fire_f_pts = net_cg772 - net_ca082

'''
x = np.arange(1850,1950)
fitT = np.polyfit(x,tg_aw310, 1)
linear_baselineT = np.poly1d(fitT) # create the linear baseline function
fitF = np.polyfit(x,net_aw310, 1)
linear_baselineF = np.poly1d(fitF) # create the linear baseline function
ctr_t_pts = (tg_bb446 - linear_baselineT(tg_bb446))/2
ctr_f_pts = net_bb446 - linear_baselineF(net_bb446)

x = np.arange(1850,1950)
fitT = np.polyfit(x,tg_ca082, 1)
linear_baselineT = np.poly1d(fitT) # create the linear baseline function
fitF = np.polyfit(x,net_ca082, 1)
linear_baselineF = np.poly1d(fitF) # create the linear baseline function
fire_t_pts = (tg_cg772 - linear_baselineT(tg_cg772))/2
fire_f_pts = net_cg772 - linear_baselineF(net_cg772)
'''


fig, ax = plt.subplots()

#Regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(fire_t_pts,fire_f_pts)
xincp = -abs(intercept)/slope
print ("fire slope, yintercept, xintercept r, p, err = ",slope,intercept,xincp,r_value,p_value,std_err)
plt.plot(fire_t_pts, intercept + slope*fire_t_pts, 'r', alpha=0.5)
ax.axline((0, intercept), slope=slope, color='r', ls='--')


slope, intercept, r_value, p_value, std_err = stats.linregress(ctr_t_pts,ctr_f_pts)
xincp = -abs(intercept)/slope
print ("ctr slope, yintercept, xintercept r, p, err = ",slope,intercept,xincp,r_value,p_value,std_err)
plt.plot(ctr_t_pts, intercept + slope*ctr_t_pts, 'b', alpha=0.5)
ax.axline((0, intercept), slope=slope, color='b', ls='--')


plt.scatter(ctr_t_pts,ctr_f_pts,marker='x', label='no fire')
plt.scatter(fire_t_pts,fire_f_pts,color='r',marker='x', label='fire')
plt.xlim(0,8)
plt.ylim(0,8)
plt.title('Climate Sensitivity')
plt.legend()
plt.xlabel('K')
plt.ylabel('TOA / W m-2')

plt.tight_layout()
plt.show()
exit()


plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Gregory.jpg')




# run this after data has been read in. Produce plots.
exec(open('/home/h01/cburton/IRIS/MECA6/MECA6_monitor_read_files.py').read())
y_cg772=y_cg772-9
y_ca082=y_ca082-9

print('Updating 4xCO2 run plots')


#-----------------------------------------------------------------
#
# 1. component radiation fluxes and TOA

xmin=1848
xmax=max(y_bb446)+30

fig = plt.figure(figsize=(12, 6))

plt.subplot(241)
plt.plot(y_cg772,netcl_cg772,'r--',linewidth=1.5,label='UKESM1 4xCO2+fire')
plt.plot(y_ca082,netcl_ca082,'k--',linewidth=1.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,netcl_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,netcl_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('NetCL')
#plt.xlabel('year')
plt.ylabel('W m-2')
#plt.legend(loc=2,fontsize=10)

plt.subplot(242)
plt.plot(y_cg772,netcf_cg772,'r--',linewidth=1.5,label='UKESM1 4xCO2+fire')
plt.plot(y_ca082,netcf_ca082,'k--',linewidth=1.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,netcf_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,netcf_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('NetCF')
#plt.xlabel('year')
#plt.ylabel('W m-2')
#plt.legend(loc=2,fontsize=10)

plt.subplot(243)
plt.plot(y_cg772,osw_cg772,'r--',linewidth=1.5,label='UKESM1 4xCO2+fire')
plt.plot(y_ca082,osw_ca082,'k--',linewidth=1.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,osw_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,osw_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('OSW')
#plt.xlabel('year')
#plt.ylabel('W m-2')
#plt.legend(loc=2,fontsize=10)

plt.subplot(244)
plt.plot(y_cg772,closw_cg772,'r--',linewidth=1.5,label='UKESM1 4xCO2+fire')
plt.plot(y_ca082,closw_ca082,'k--',linewidth=1.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,closw_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,closw_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('CLOSW')
#plt.xlabel('year')
#plt.ylabel('W m-2')
#plt.legend(loc=2,fontsize=10)

plt.subplot(245)
plt.plot(y_cg772,swcf_cg772,'r--',linewidth=1.5,label='UKESM1 4xCO2+fire')
plt.plot(y_ca082,swcf_ca082,'k--',linewidth=1.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,swcf_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,swcf_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('SWCF')
plt.xlabel('year')
plt.ylabel('W m-2')
plt.legend(loc=1,fontsize=6)

plt.subplot(246)
plt.plot(y_cg772,olw_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,olw_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,olw_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,olw_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('OLW')
plt.xlabel('year')
#plt.ylabel('W m-2')
#plt.legend(loc=2,fontsize=10)

plt.subplot(247)
plt.plot(y_cg772,clolw_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,clolw_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,clolw_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,clolw_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('CLOLW')
plt.xlabel('year')
plt.ylabel('W m-2')
#plt.legend(loc=2,fontsize=10)

plt.subplot(248)
plt.plot(y_cg772,lwcf_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,lwcf_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,lwcf_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,lwcf_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('LWCF')
plt.xlabel('year')
#plt.ylabel('W m-2')
#plt.legend(loc=2,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/TOA_cpts_4xCO2.jpg')
#plt.show()

fig = plt.figure(figsize=(6, 6))
plt.plot(y_cg772,net_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,net_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,net_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,net_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.plot([0,3e3],[0,0],'k--',linewidth=.5)
plt.xlim(xmin,xmax)
plt.title('Net TOA')
plt.xlabel('year')
plt.ylabel('W m-2')
plt.legend(loc=1,fontsize=12)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/TOA_4xCO2.jpg')
#plt.show()

#-----------------------------------------------------------------
#
# 2. Global T & precip

fig = plt.figure(figsize=(6, 6))

plt.subplot(211)
plt.plot(y_cg772,tg_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,tg_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,tg_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,tg_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Global T')
#plt.xlabel('year')
plt.ylabel('K')
plt.legend(loc=5,fontsize=8)

plt.subplot(212)
plt.plot(y_cg772,pg_cg772*86400.,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,pg_ca082*86400.,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,pg_bb446*86400.,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,pg_aw310*86400.,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Global P')
plt.xlabel('year')
plt.ylabel('mm day-1')
#plt.legend(loc=2,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Glob_T_ppn_4xCO2.jpg')
#plt.show()

# 2.1 split land and ocean

fig = plt.figure(figsize=(8, 8))

plt.subplot(221)
plt.plot(y_cg772,tl_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,tl_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,tl_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,tl_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Land T')
#plt.xlabel('year')
plt.ylabel('K')
plt.legend(loc=5,fontsize=8)

plt.subplot(222)
plt.plot(y_cg772,pl_cg772*86400.,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,pl_ca082*86400.,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,pl_bb446*86400.,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,pl_aw310*86400.,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Land P')
#plt.xlabel('year')
plt.ylabel('mm day-1')
#plt.legend(loc=2,fontsize=10)

plt.subplot(223)
plt.plot(y_cg772,to_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,to_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,to_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,to_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Ocean T')
plt.xlabel('year')
plt.ylabel('K')
plt.legend(loc=2,fontsize=8)

plt.subplot(224)
plt.plot(y_cg772,po_cg772*86400.,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,po_ca082*86400.,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,po_bb446*86400.,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,po_aw310*86400.,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Ocean P')
plt.xlabel('year')
plt.ylabel('mm day-1')
#plt.legend(loc=2,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Land_ocean_T_ppn_4xCO2.jpg')
#plt.show()

#-----------------------------------------------------------------
#
# 2b. Gregory plots

# UKESM1 control
print('UKESM1 removing average global temperature and TOA from control run of: ', np.mean(tg_aw310), np.mean(net_aw310))
ctr_t_pts = tg_bb446 - np.mean(tg_aw310)
ctr_f_pts = net_bb446 - np.mean(net_aw310)

fig = plt.figure(figsize=(6, 6))
plt.plot([0,8],[7,0],'#010101',linewidth=0.2)
plt.plot([0,10],[7,0],'#010101',linewidth=0.2)
plt.plot([0,12],[7,0],'#010101',linewidth=0.2)
plt.scatter(ctr_t_pts,ctr_f_pts,marker='x')
plt.xlim(0,12)
plt.ylim(0,7.2)
plt.title('UKESM1')
plt.xlabel('K')
plt.ylabel('TOA / W m-2')
#plt.legend(fontsize=6)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/UKESM_Gregory.jpg')
#plt.show()

# UKESM Fire
print('Fire removing average global temperature and TOA from control run of: ', np.mean(tg_ca082), np.mean(net_ca082))
fire_t_pts = tg_cg772 - np.mean(tg_ca082)
fire_f_pts = net_cg772 - np.mean(net_ca082)

fig = plt.figure(figsize=(6, 6))
plt.plot([0,8],[7,0],'#010101',linewidth=0.2)
plt.plot([0,10],[7,0],'#010101',linewidth=0.2)
plt.plot([0,12],[7,0],'#010101',linewidth=0.2)
plt.scatter(fire_t_pts,fire_f_pts,marker='x')
plt.xlim(0,12)
plt.ylim(0,7.2)
plt.title('UKESM Fire')
plt.xlabel('K')
plt.ylabel('TOA / W m-2')
#plt.legend(fontsize=6)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Fire_Gregory.jpg')
#plt.show()


# UKESM
print('Fire removing average global temperature and TOA from control run of: ', np.mean(tg_ca082), np.mean(net_ca082))

fig = plt.figure(figsize=(6, 6))
plt.plot([0,8],[7,0],'#010101',linewidth=0.2)
plt.plot([0,10],[7,0],'#010101',linewidth=0.2)
plt.plot([0,12],[7,0],'#010101',linewidth=0.2)

#Regression line (short method)
slope, intercept, r_value, p_value, std_err = stats.linregress(fire_t_pts,fire_f_pts)
print ("fire slope, intercept, r, p, err = ",slope,intercept,r_value,p_value,std_err)
plt.plot(fire_t_pts, intercept + slope*fire_t_pts, 'r', alpha=0.5)

slope, intercept, r_value, p_value, std_err = stats.linregress(ctr_t_pts,ctr_f_pts)
print ("ctr slope, intercept, r, p, err = ",slope,intercept,r_value,p_value,std_err)
plt.plot(ctr_t_pts, intercept + slope*ctr_t_pts, 'b', alpha=0.5)

plt.scatter(ctr_t_pts,ctr_f_pts,marker='x', label='no fire')
plt.scatter(fire_t_pts,fire_f_pts,color='r',marker='x', label='fire')
plt.xlim(0,12)
plt.ylim(0,7.2)
plt.title('Climate Sensitivity')
plt.legend()
plt.xlabel('K')
plt.ylabel('TOA / W m-2')
#plt.legend(fontsize=6)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Gregory.jpg')
#plt.show()


#-----------------------------------------------------------------
#
# 4. Global soil and veg carbon

fig = plt.figure(figsize=(6, 6))

plt.subplot(211)
plt.plot(y_cg772,cv_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,cv_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,cv_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,cv_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Veg Carbon')
#plt.xlabel('year')
plt.ylabel('PgC')
plt.legend(loc=2,fontsize=10)

plt.subplot(212)
plt.plot(y_cg772,cs_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,cs_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,cs_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,cs_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Soil Carbon')
plt.xlabel('year')
plt.ylabel('PgC')
#plt.legend(loc=2,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Veg_soil_carbon_4xCO2.jpg')
#plt.show()

# 4.1 split cSoil components

fig = plt.figure(figsize=(8, 8))

plt.subplot(221)
plt.plot(y_cg772,dpm_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,dpm_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,dpm_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,dpm_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('DPM')
#plt.xlabel('year')
plt.ylabel('PgC')
#plt.legend(loc=2,fontsize=10)

plt.subplot(222)
plt.plot(y_cg772,rpm_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,rpm_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,rpm_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,rpm_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('RPM')
#plt.xlabel('year')
#plt.ylabel('PgC')
#plt.legend(loc=2,fontsize=10)

plt.subplot(223)
plt.plot(y_cg772,bio_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,bio_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,bio_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,bio_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('BIO')
plt.xlabel('year')
plt.ylabel('PgC')
#plt.legend(loc=2,fontsize=10)

plt.subplot(224)
plt.plot(y_cg772,hum_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,hum_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,hum_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,hum_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('HUM')
plt.xlabel('year')
#plt.ylabel('PgC')
plt.legend(loc=6,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Soil_carbon_pools_4xCO2.jpg')
#plt.show()

#-----------------------------------------------------------------
#
# 4b. Global soil and veg carbon drifts

fig = plt.figure(figsize=(6, 8))

plt.subplot(311)
plt.plot(y_cg772-9,cv_cg772-cv_cg772[0],'k--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_bb446,cv_bb446-cv_bb446[0],'k',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot([0,3e3],[0,0],'k--',linewidth=.5)
plt.plot([1850,2850],[0,100],'r--')
plt.plot([1850,2850],[0,-100],'r--')
plt.xlim(xmin,xmax)
plt.ylim(-10,10)
plt.title('PI-Ctl Veg Carbon Drift')
#plt.xlabel('year')
plt.ylabel('PgC')
plt.legend(loc=2,fontsize=10)

plt.subplot(312)
plt.plot(y_cg772-9,cs_cg772-cs_cg772[0],'k--',linewidth=.5)
plt.plot(y_bb446,cs_bb446-cs_bb446[0],'k',linewidth=1.5)
plt.plot([0,3e3],[0,0],'k--',linewidth=.5)
plt.plot([1850,2850],[0,100],'r--',label='acceptable drift')
plt.plot([1850,2850],[0,-100],'r--')
plt.xlim(xmin,xmax)
plt.ylim(-10,10)
plt.title('PI-Ctl Soil Carbon Drift')
#plt.xlabel('year')
plt.ylabel('PgC')
plt.legend(loc=2,fontsize=10)

plt.subplot(313)
plt.plot(y_cg772-9,cs_cg772-cs_cg772[0]+cv_cg772-cv_cg772[0],'k--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_bb446,cs_bb446-cs_bb446[0]+cv_bb446-cv_bb446[0],'k',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot([0,3e3],[0,0],'k--',linewidth=.5)
plt.plot([1850,2850],[0,100],'r--')
plt.plot([1850,2850],[0,-100],'r--')
plt.xlim(xmin,xmax)
plt.ylim(-10,10)
plt.title('Total Terrest. Carbon Drift')
plt.xlabel('year')
plt.ylabel('PgC')
#plt.legend(loc=2,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Soil_carbon_pools_drift_4xCO2.jpg')
#plt.show()


#-----------------------------------------------------------------
#
# 5. Carbon fluxes

fig = plt.figure(figsize=(8, 8))

plt.subplot(221)
plt.plot(y_cg772,npp_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,npp_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,npp_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,npp_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('NPP')
#plt.xlabel('year')
plt.ylabel('PgC yr-1')
#plt.legend(loc=2,fontsize=10)

plt.subplot(222)
plt.plot(y_cg772,gpp_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,gpp_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,gpp_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,gpp_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('GPP')
#plt.xlabel('year')
#plt.ylabel('PgC')
#plt.legend(loc=2,fontsize=10)

plt.subplot(223)
plt.plot(y_cg772,rs_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,rs_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,rs_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,rs_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Soil resp.')
plt.xlabel('year')
plt.ylabel('PgC yr-1')
#plt.legend(loc=2,fontsize=10)

plt.subplot(224)
plt.plot(y_cg772,nep_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,nep_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,nep_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,nep_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.plot([0,3e3],[0,0],'k--',linewidth=.5)
plt.xlim(xmin,xmax)
plt.title('NEP')
plt.xlabel('year')
#plt.ylabel('PgC')
plt.legend(loc=1,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Carbon_fluxes_4xCO2.jpg')
#plt.show()

#
# 5b. Carbon use efficiency

fig = plt.figure(figsize=(8, 8))

plt.subplot(221)
plt.plot(y_cg772,cue_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,cue_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,cue_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,cue_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('CUE: Global')
#plt.xlabel('year')
#plt.ylabel('')
plt.legend(loc=6,fontsize=10)

plt.subplot(222)
plt.plot(y_cg772,cue_trop_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,cue_trop_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,cue_trop_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,cue_trop_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('CUE: Tropics')
#plt.xlabel('year')
#plt.ylabel('')
#plt.legend(loc=2,fontsize=10)

plt.subplot(223)
plt.plot(y_cg772,cue_temp_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,cue_temp_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,cue_temp_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,cue_temp_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('CUE: Temperate')
plt.xlabel('year')
#plt.ylabel('')
#plt.legend(loc=2,fontsize=10)

plt.subplot(224)
plt.plot(y_cg772,cue_boreal_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,cue_boreal_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,cue_boreal_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,cue_boreal_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.plot([0,3e3],[0,0],'k--',linewidth=.5)
plt.xlim(xmin,xmax)
plt.title('CUE: Boreal')
plt.xlabel('year')
#plt.ylabel('')
#plt.legend(loc=2,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/CUE_4xCO2.jpg')
#plt.show()

#-----------------------------------------------------------------
#
# 6. Ocean CO2 flux

fig = plt.figure(figsize=(6, 6))

plt.plot(y_cg772,ocnflx_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,ocnflx_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,ocnflx_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,ocnflx_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.plot([0,3e3],[0,0],'k--',linewidth=.5)
plt.xlim(xmin,xmax)
plt.title('air-to-sea CO2 flux')
plt.xlabel('year')
plt.ylabel('PgC yr-1')
plt.legend(loc=1,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Ocean_carbon_4xCO2.jpg')
#plt.show()

#-----------------------------------------------------------------
#
# 6b. Total CO2 flux to Atm

fig = plt.figure(figsize=(6, 6))

plt.plot(y_aw310,-ocnflx_aw310,'k:',linewidth=.5,label='ocean')
plt.plot(y_bb446,-ocnflx_bb446,'r:',linewidth=.5)

plt.plot(y_aw310,-nep_aw310,'k--',linewidth=.5,label='land')
plt.plot(y_bb446,-nep_bb446,'r--',linewidth=.5)

plt.plot(y_aw310,-ocnflx_aw310-nep_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.plot(y_bb446,-ocnflx_bb446-nep_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')

plt.plot([0,3e3],[0,0],'k--',linewidth=.5)
plt.xlim(xmin,xmax)
plt.title('Net CO2 flux to Atmos')
plt.xlabel('year')
plt.ylabel('PgC yr-1')
plt.legend(loc=4,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Atm_net_CO2_flux_4xCO2.jpg')
#plt.show()

#-----------------------------------------------------------------
#
# 7. Ice frac

fig = plt.figure(figsize=(6, 6))

plt.subplot(211)
plt.plot(y_cg772,ice_nh_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,ice_nh_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,ice_nh_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,ice_nh_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('North Hemi. ice-frac')
#plt.xlabel('year')
plt.ylabel('10^12 m2')
plt.legend(loc=1,fontsize=10)

plt.subplot(212)
plt.plot(y_cg772,ice_sh_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,ice_sh_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,ice_sh_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,ice_sh_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('South Hemi. ice-frac')
plt.xlabel('year')
plt.ylabel('10^12 m2')
#plt.legend(loc=2,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Ice_4xCO2.jpg')
#plt.show()

#-----------------------------------------------------------------
#
# 9. Veg fractions

fig = plt.figure(figsize=(8, 8))

plt.subplot(221)
plt.plot(y_cg772,tree_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,tree_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,tree_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,tree_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Global tree frac')
#plt.xlabel('year')
plt.ylabel('frac')
plt.legend(loc=2,fontsize=10)

plt.subplot(222)
plt.plot(y_cg772,grass_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,grass_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,grass_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,grass_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Global grass frac')
#plt.xlabel('year')
#plt.ylabel('frac')
#plt.legend(loc=2,fontsize=10)

plt.subplot(223)
plt.plot(y_cg772,bare_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,bare_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,bare_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,bare_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Global bare frac')
plt.xlabel('year')
plt.ylabel('frac')
#plt.legend(loc=2,fontsize=10)

plt.subplot(224)
plt.plot(y_cg772,amaz_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,amaz_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,amaz_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,amaz_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Amazon tree frac')
plt.xlabel('year')
#plt.ylabel('frac')
#plt.legend(loc=2,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Veg_fracs_4xCO2.jpg')
#plt.show()

#-----------------------------------------------------------------
#
# 11. Land use pools/fluxes / fractions

fig = plt.figure(figsize=(8, 8))

plt.subplot(221)
plt.plot(y_cg772,luf_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,luf_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,luf_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,luf_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Fast turnover product pool')
#plt.xlabel('year')
plt.ylabel('PgC')
plt.legend(loc=2,fontsize=10)

plt.subplot(222)
plt.plot(y_cg772,lum_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,lum_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,lum_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,lum_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Medium turnover product pool')
#plt.xlabel('year')
#plt.ylabel('PgC')
#plt.legend(loc=2,fontsize=10)

plt.subplot(223)
plt.plot(y_cg772,lus_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,lus_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,lus_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,lus_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Slow turnover product pool')
plt.xlabel('year')
plt.ylabel('PgC')
#plt.legend(loc=2,fontsize=10)

plt.subplot(224)
plt.plot(y_cg772,luflx_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,luflx_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,luflx_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,luflx_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('LU flux to atmos')
plt.xlabel('year')
plt.ylabel('PgC yr-1')
#plt.legend(loc=2,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/LU_4xCO2.jpg')
#plt.show()

# LU crop and pasture fracs: C3 vs C4 for trop/ex-tropics
fig = plt.figure(figsize=(6, 6))

plt.subplot(211)
plt.plot(y_bb446,luflx_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,luflx_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('LU flux to atmos')
#plt.xlabel('year')
plt.ylabel('PgC yr-1')
plt.legend(loc=2,fontsize=10)

plt.subplot(212)
plt.plot(y_bb446,hvflx_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,hvflx_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Harvest flux to atmos')
plt.xlabel('year')
plt.ylabel('PgC yr-1')
#plt.legend(loc=2,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/LU_Harv_flx_4xCO2.jpg')


# LU and harvest flux
fig = plt.figure(figsize=(8, 8))

plt.subplot(221)
plt.plot(y_bb446,lu_c3c_extrop_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2 C3')
plt.plot(y_aw310,lu_c3c_extrop_aw310,'k',linewidth=1.5,label='UKESM1 ctl C3')
plt.plot(y_bb446,lu_c4c_extrop_bb446,'r--',linewidth=1.5,label='UKESM1 4xCO2 C4')
plt.plot(y_aw310,lu_c4c_extrop_aw310,'k--',linewidth=1.5,label='UKESM1 ctl C4')
plt.xlim(xmin,xmax)
plt.title('Crop fractions')
#plt.xlabel('year')
plt.ylabel('extra-tropics')
plt.legend(loc=2,fontsize=10)

plt.subplot(222)
plt.plot(y_bb446,lu_c3p_extrop_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2 C3')
plt.plot(y_aw310,lu_c3p_extrop_aw310,'k',linewidth=1.5,label='UKESM1 ctl C3')
plt.plot(y_bb446,lu_c4p_extrop_bb446,'r--',linewidth=1.5,label='UKESM1 4xCO2 C4')
plt.plot(y_aw310,lu_c4p_extrop_aw310,'k--',linewidth=1.5,label='UKESM1 ctl C4')
plt.xlim(xmin,xmax)
plt.title('Pasture fractions')
#plt.xlabel('year')
plt.ylabel('extra-tropics')
#plt.legend(loc=2,fontsize=10)

plt.subplot(223)
plt.plot(y_bb446,lu_c3c_trop_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2 C3')
plt.plot(y_aw310,lu_c3c_trop_aw310,'k',linewidth=1.5,label='UKESM1 ctl C3')
plt.plot(y_bb446,lu_c4c_trop_bb446,'r--',linewidth=1.5,label='UKESM1 4xCO2 C4')
plt.plot(y_aw310,lu_c4c_trop_aw310,'k--',linewidth=1.5,label='UKESM1 ctl C4')
plt.xlim(xmin,xmax)
#plt.title('Crop fractions')
plt.xlabel('year')
plt.ylabel('tropics')
#plt.legend(loc=2,fontsize=10)

plt.subplot(224)
plt.plot(y_bb446,lu_c3p_trop_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2 C3')
plt.plot(y_aw310,lu_c3p_trop_aw310,'k',linewidth=1.5,label='UKESM1 ctl C3')
plt.plot(y_bb446,lu_c4p_trop_bb446,'r--',linewidth=1.5,label='UKESM1 4xCO2 C4')
plt.plot(y_aw310,lu_c4p_trop_aw310,'k--',linewidth=1.5,label='UKESM1 ctl C4')
plt.xlim(xmin,xmax)
#plt.title('Pasture fractions')
plt.xlabel('year')
plt.ylabel('tropics')
#plt.legend(loc=2,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/LU_fracs_4xCO2.jpg')


#-----------------------------------------------------------------
#
# 12. CH4 emissions / wetland frac

fig = plt.figure(figsize=(6, 6))

plt.subplot(211)
plt.plot(y_cg772,ch4em_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,ch4em_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,ch4em_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,ch4em_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('CH4 emissions')
#plt.ylim(0,300)
#plt.xlabel('year')
plt.ylabel('TgCH4 yr-1')
#plt.legend(loc=2,fontsize=10)

plt.subplot(212)
plt.plot(y_cg772,wetl_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,wetl_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,wetl_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,wetl_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Wetland fraction')
plt.xlabel('year')
plt.ylabel('frac')
plt.legend(loc=1,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/CH4_Wetland_4xCO2.jpg')


#-----------------------------------------------------------------
#
# 13. Burnt Area / Fire emissions

fig = plt.figure(figsize=(6, 6))

plt.subplot(211)
plt.plot(y_cg772,BA_cg772,'r--',linewidth=.5,label='UKESM1 4XCO2+fire')
plt.plot(y_ca082,BA_ca082,'k--',linewidth=.5,label='UKESM1 ctl+fire')
plt.plot(y_bb446,BA_bb446,'r',linewidth=1.5,label='UKESM1 4xCO2')
plt.plot(y_aw310,BA_aw310,'k',linewidth=1.5,label='UKESM1 ctl')
plt.xlim(xmin,xmax)
plt.title('Burnt Area')
#plt.ylim(0,300)
#plt.xlabel('year')
plt.ylabel('Mkm$^2$')
#plt.legend(loc=2,fontsize=10)

plt.subplot(212)
plt.plot(y_cg772,Femiss_cg772,'r',linewidth=.5,label='4XCO2+fire C')
plt.plot(y_ca082,Femiss_ca082,'g',linewidth=.5,label='ctl+fire C')
plt.plot(y_bb446,Femiss_bb446,'k.',linewidth=1.5,label='4xCO2 C')
plt.plot(y_aw310,Femiss_aw310,'b',linewidth=1.5,label='ctl C')

plt.plot(y_cg772,FemissDPM_cg772,'r-.',linewidth=.5,label='4XCO2+fire DPM')
plt.plot(y_ca082,FemissDPM_ca082,'g-.',linewidth=.5,label='ctl+fire DPM')
plt.plot(y_bb446,FemissDPM_bb446,'k-.',linewidth=1.5,label='4xCO2 DPM')
plt.plot(y_aw310,FemissDPM_aw310,'b-.',linewidth=1.5,label='ctl DPM')

plt.plot(y_cg772,FemissRPM_cg772,'r:',linewidth=.5,label='4XCO2+fire RPM')
plt.plot(y_ca082,FemissRPM_ca082,'g:',linewidth=.5,label='ctl+fire RPM')
plt.plot(y_bb446,FemissRPM_bb446,'k:',linewidth=1.5,label='4xCO2 RPM')
plt.plot(y_aw310,FemissRPM_aw310,'b:',linewidth=1.5,label='ctl RPM')

plt.plot(y_cg772,FemissT_cg772,'r--',linewidth=.5,label='4XCO2+fire Total')
plt.plot(y_ca082,FemissT_ca082,'g--',linewidth=.5,label='ctl+fire Total')
plt.plot(y_bb446,FemissT_bb446,'k--',linewidth=1.5,label='4xCO2 Total')
plt.plot(y_aw310,FemissT_aw310,'b--',linewidth=1.5,label='ctl Total ')

plt.xlim(xmin,xmax)
plt.title('Fire Emissions')
plt.xlabel('year')
plt.ylabel('GtC')
plt.legend(loc=1,fontsize=10)

plt.tight_layout()
plt.savefig('/home/h01/cburton/public_html/work/MECA6/Output/jpgs/Fire_4xCO2.jpg')











