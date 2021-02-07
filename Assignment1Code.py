import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy_financial as npf
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

# Load bond data
bond_df = pd.read_csv('SelectedCANGovBonds.csv', index_col=0)

'''
Question 4
'''
# list of all the trading days from Jan 18 - Jan 29
all_days = [18, 19, 20, 21, 22, 25, 26, 27, 28, 29]

# list of the bond maturities for the 11 selected bonds, in increasing order
# inserted 'NaN' in the space where there's a missing bond (no bond maturing in fall 2023)
all_mats = ['2021-05-01', '2021-11-01', '2022-05-01', '2022-11-01', '2023-03-01', '2023-06-01', 'NaN', '2024-03-01', '2024-09-01',
            '2025-03-01', '2025-09-01', '2026-03-01']

# initialize dictionary for ytm values
ytm_dict = {}

# initialize dictionary for spot rate values
spot_dict = {}

# Calculate the yield and spot curves for each day and store the data in their respective dictionaries
for day in all_days:
    # convert the date to datetime so it can be manipulated later
    date = pd.to_datetime('2021-01-'+str(day))
    # get a list of all the bond data on the given date
    all_bonds_on_date = bond_df.loc['2021-01-'+str(day)]
    # iterate through each of the bonds to calculate their yield/the spot rate
    # for loop iterates through the bonds in order of increasing maturity
    for i, mat_date in enumerate(all_mats):
        # if at the point in the list of bonds where there's a missing bond
        if mat_date == 'NaN':
            # extrapolate the spot rate at T = 2.7
            T = 2.7

            # use linear extrapolation to calculate spot rate for missing bond
            x_data = spot_dict['2021-01-'+str(day)].index.values
            y_data = spot_dict['2021-01-'+str(day)].values
            f_spot_linear = interp1d(x_data, y_data, fill_value='extrapolate')
            spot_r = f_spot_linear(T)
            # store the value of the spot rate at time T in the dictionary at the given date
            spot_dict['2021-01-' + str(day)].loc[T] = spot_r
            continue

        # extract data on bond i with maturity mat_date from the list of all bond data on the given date
        bond = all_bonds_on_date[all_bonds_on_date['maturity date'] == str(mat_date)]
        # record coupon value and clean price for bond i with maturity mat_date
        coupon = bond.loc['2021-01-'+str(day), 'coupon']
        clean_price = bond.loc['2021-01-'+str(day), 'historical close price']

        # Calculate dirty price
        # convert maturity date to datetime so it can be manipulated
        mat_date = pd.to_datetime(mat_date)
        if i >= 5:
            # since there are 2 bonds with maturities between 2 and 2.5yrs,
            # need to subtract i*6 months from the bond maturity date for bonds 5 to 11
            # to get the date of the last coupon payment (i.e. for bond 5 with maturity June 2023,
            # last coupon payment date = Jun 2023 - 5*0.5yrs = December 2020)
            last_coupon_payment = mat_date + relativedelta(months= -i*6)
        else:
            # need to subtract (i+1)*6 months from the bond maturity date for bonds 0 to 4
            # to get the date of the last coupon payment (i.e. for bond 0 with maturity May 2021,
            # last coupon payment date = May 2021 - (0+1)*0.5yrs = Nov 2020)
            last_coupon_payment = mat_date + relativedelta(months= -(i+1)*6)
        # n: number of days since last coupon payment
        n = date - last_coupon_payment
        # accrued interest = n/365*(annual coupon rate * 100)
        accrued_int = (n.days/365)*(coupon*100)
        # dirty price = accrued interest + clean price
        dirty_price = accrued_int + clean_price

        # Calculate the time to maturity
        T = (mat_date - date).days/365

        '''
        Calculate yield to maturity
        '''
        # initialize list of cashflows
        cashflows = [-dirty_price]
        # initialize coupon date to the next possible coupon date
        coupon_date = last_coupon_payment + relativedelta(months=+6)
        # while the coupon date is before maturity date
        while coupon_date < mat_date:
            # add the coupon payment to the list of cashflows
            cashflows = cashflows + [(coupon*100)/2]
            # increment coupon_date to the date of the next possible coupon payment
            coupon_date = coupon_date + relativedelta(months=+6)
        # append the final payment (face value + coupon) to the cashflow list
        cashflows = cashflows + [100 + (coupon*100)/2]
        # Use IRR function to calculate 6-month interest rate
        r_6m = npf.irr(cashflows)
        # Calculate yield to maturity
        ytm = 2*r_6m

        # record ytm
        # if i == 0 initialize series in ytm_dict[date]
        if i == 0:
            ytm_dict['2021-01-'+str(day)] = pd.Series(dtype='float64')
        ytm_dict['2021-01-'+str(day)].loc[T] = ytm

        '''
        Calculate spot rate
        '''

        # Since bond 0 is the 5 month bond, use the equation for the spot rate if T < 6
        if i == 0:
            # Spot rate equation if T < 6 months
            spot_r = - np.log(dirty_price/(100 + (coupon*100)/2))/T
            # initialize series in spot_dict[date]
            spot_dict['2021-01-' + str(day)] = pd.Series(dtype='float64')
            # store the value of the spot rate at time T in the dictionary at the given date
            spot_dict['2021-01-' + str(day)].loc[T] = spot_r

        # Since all other bonds have maturities > 6 months, for all other cases use bootstrapping method
        else:
            # discount all coupon payments that are made before the maturity date and then sum them all
            # initialize a variable to hold the sum of all the discounted coupon payments
            sum_discounted_payments = 0
            # initialize coupon date to the next possible coupon date
            coupon_date = last_coupon_payment + relativedelta(months=+6)
            # while the coupon date is before maturity date
            while coupon_date < mat_date:
                # t_j is the time between the current date and the coupon date, measured in years
                t_j = (coupon_date - date).days/365

                # if the time of the coupon payment (t_j) is before the maturity date of bond 0,
                # cannot interpolate to find the spot rate at t_j because there is no data in the dataset before
                # the maturity date of bond 0. Thus, assume constant spot rate
                # to avoid extrapolation error, and use the spot rate at the time
                # of the earliest bond maturity date (so the spot rate obtained from bond 0)
                if t_j <= spot_dict['2021-01-' + str(day)].index[0]:
                     spot_t_j = spot_dict['2021-01-' + str(day)].iloc[0]
                else:
                    # use linear interpolation to find spot rate at time t_j
                    x_data = spot_dict['2021-01-' + str(day)].index.values
                    y_data = spot_dict['2021-01-' + str(day)].values
                    f_spot_linear = interp1d(x_data, y_data)
                    spot_t_j = f_spot_linear(t_j)

                # increment the sum of discounted coupon payments by the additional discounted coupon payment
                sum_discounted_payments = sum_discounted_payments + ((coupon*100)/2)*np.exp(-spot_t_j*t_j)

                # increment coupon_date to the date of the next possible coupon payment
                coupon_date = coupon_date + relativedelta(months=+6)

            # use the bootstrapping equation to calculate the spot rate at time T
            spot_r = - np.log((dirty_price - sum_discounted_payments)/(100 + (coupon*100)/2))/T
            # store the value of the spot rate at time T in the dictionary at the given date
            spot_dict['2021-01-' + str(day)].loc[T] = spot_r

'''
Calculate forward rate
'''
# list of bonds (in increasing order) used to calculate 1yr forward curve (maturities: 1yr - 5yr)
fr_mats = ['2023-03-01', '2023-06-01',  '2024-03-01', '2024-09-01',
            '2025-03-01', '2025-09-01', '2026-03-01']

# initialize dictionary for forward rate values
fr_dict = {}

# Calculate the forward curves for each day and store the data in fr_dict
for day in all_days:
    # convert the date to datetime so it can be manipulated later
    date = pd.to_datetime('2021-01-'+str(day))
    # get a list of all the bond data on the given date
    all_bonds_on_date = bond_df.loc['2021-01-'+str(day)]

    # use linear interpolation to find the 1yr spot rate
    t_m = 1
    x_data = spot_dict['2021-01-' + str(day)].index.values
    y_data = spot_dict['2021-01-' + str(day)].values
    f_spot_linear = interp1d(x_data, y_data)
    # spot_m = 1 yr spot rate
    spot_m = f_spot_linear(t_m)

    # generate a list of the maturity times for all bonds that mature in 2+ years
    bond_mat_times = [t for t in spot_dict['2021-01-' + str(day)].index.values if t >= 2]

    # iterate through each of the bonds to calculate the forward rate at each time t_n
    # for loop iterates through the bonds in order of increasing maturity, t_n
    for i, t_n in enumerate(bond_mat_times):
        # use linear interpolation to find the spot rate at t_n
        spot_n = f_spot_linear(t_n)

        # calculate forward rate
        # forward rate = [(1 + spot_n)^t_n/(1 + spot_m)^t_m] - 1, where n > m
        fr = ((1 + spot_n)**t_n/(1 + spot_m)**t_m) - 1

        # record forward rate
        # if i == 0 initialize dataframe in fr_dict[date]
        if i == 0:
            fr_dict['2021-01-'+str(day)] = pd.Series(dtype='float64')
        fr_dict['2021-01-'+str(day)].loc[t_n] = fr



'''
Plot yield curves
'''
# Initialize axis to plot the figures
fig = plt.figure()
ax = fig.add_subplot(111)

# create a set of colours to use to plot the curves
pal = ['#00CED1', '#FF8C00', '#006400', '#1E90FF', '#9400D3', '#DC143C',
        '#FA8072', '#FFD700', '#00008B', '#3CB371']

# Iterate through each day of data and plot the yield curve
for i, day in enumerate(all_days):
    date = '2021-01-'+str(day)

    # Compute cubic spline interpolant for ytm data
    x_ytm_data = ytm_dict[date].index.values
    y_ytm_data = ytm_dict[date].values
    f_ytm_cubic = CubicSpline(x_ytm_data, y_ytm_data)

    ytm_x_pts = np.linspace(0, 5, 100)

    # Plot the cubic spline interpolant for ytm
    ax.plot(ytm_x_pts, f_ytm_cubic(ytm_x_pts), color=pal[i], ls='-', label='Jan '+str(day))

plt.legend(loc=0)
plt.title("Yield curve from Jan 18 - Jan 29 2021 ")
plt.xlabel("Year")
plt.ylabel("Yield-to-Maturity")
plt.grid(b=True, which='major', color='#c9c9c9', linestyle='-')
plt.show()

'''
Plot spot curves
'''
# Initialize axis to plot the figures
fig = plt.figure()
ax = fig.add_subplot(111)
# Iterate through each day of data and plot the yield curve
for i, day in enumerate(all_days):
    date = '2021-01-'+str(day)

    # Compute cubic spline interpolant for spot rate data
    x_spot_data = spot_dict[date].index.values
    y_spot_data = spot_dict[date].values
    f_spot_cubic = CubicSpline(x_spot_data, y_spot_data)

    spot_x_pts = np.linspace(1, 5, 100)

    # Plot the cubic spline interpolant for spot rate
    ax.plot(spot_x_pts, f_spot_cubic(spot_x_pts), color=pal[i], ls='-', label='Jan '+str(day))

plt.legend(loc=0)
plt.title("Spot curve from Jan 18 - Jan 29 2021 ")
plt.xlabel("Year")
plt.ylabel("Spot rate")
plt.grid(b=True, which='major', color='#c9c9c9', linestyle='-')
plt.show()

'''
Plot 1yr forward curves
'''
# Initialize axis to plot the figures
fig = plt.figure()
ax = fig.add_subplot(111)

# Iterate through each day of data and plot the yield curve
for i, day in enumerate(all_days):
    date = '2021-01-'+str(day)

    # Compute cubic spline interpolant for forward rate data
    x_fr_data = fr_dict[date].index.values
    y_fr_data = fr_dict[date].values
    f_fr_cubic = CubicSpline(x_fr_data, y_fr_data)

    fr_x_pts = np.linspace(2, 5, 100)

    # Plot the cubic spline interpolant for forward rate
    ax.plot(fr_x_pts, f_fr_cubic(fr_x_pts), color=pal[i], ls='-', label='Jan '+str(day))

plt.legend(loc=0)
plt.title("1-Year forward curve from Jan 18 - Jan 29 2021 ")
plt.xlabel("Year")
plt.ylabel("1-Year forward rate")
plt.grid(b=True, which='major', color='#c9c9c9', linestyle='-')
plt.show()

'''
Question 5
'''
# Initialize dataframes to hold timeseries data for 1-5yr rates for each day
q5_ytm = pd.DataFrame(columns=[1, 2, 3, 4, 5], dtype='float64')
q5_fr = pd.DataFrame(columns=[2, 3, 4, 5], dtype='float64')

# Iterate through each day to collect the ytm and forward rate data
for day in all_days:
    date = '2021-01-' + str(day)

    # Compute cubic spline interpolant for ytm data
    x_ytm_data = ytm_dict[date].index.values
    y_ytm_data = ytm_dict[date].values
    f_ytm_cubic = CubicSpline(x_ytm_data, y_ytm_data)

    # Compute cubic spline interpolant for forward rate data
    x_fr_data = fr_dict[date].index.values
    y_fr_data = fr_dict[date].values
    f_fr_cubic = CubicSpline(x_fr_data, y_fr_data)

    # Record the ytm and forward rate for each year
    for year in range(1, 6, 1):
        if year == 1:
            q5_ytm.loc[day, year] = f_ytm_cubic(year)
        else:
            q5_ytm.loc[day, year] = f_ytm_cubic(year)
            q5_fr.loc[day, year] = f_fr_cubic(year)

# Calculate the log returns
q5_ytm_ln = np.log(q5_ytm/q5_ytm.shift(1))
q5_fr_ln = np.log(q5_fr/q5_fr.shift(1))

# Calculate covariance matrices
ytm_cov = q5_ytm_ln.cov()
print("Covariance Matrix for Daily Log-Returns of Yield:")
print(ytm_cov)
print("")
fr_cov = q5_fr_ln.cov()
print("Covariance Matrix for Daily Log-Returns of Forward Rates:")
print(fr_cov)
print("")

'''
Question 6
'''
# apply PCA to the yield covariance matrix and record the eigenvectors and eigenvalues
ytm_pca = PCA()
ytm_pca_results = ytm_pca.fit(ytm_cov)
ytm_eigenvectors = ytm_pca_results.components_
ytm_eigenvalues = ytm_pca_results.explained_variance_

print("Eigenvectors of Yield Covariance Matrix:")
print(np.round(ytm_eigenvectors, 3))
print("")
print("Eigenvalues of Yield Covariance Matrix:")
print(ytm_eigenvalues)
print("")

# apply PCA to the forward rate covariance matrix and record the eigenvectors and eigenvalues
fr_pca = PCA()
fr_pca_results = fr_pca.fit(fr_cov)
fr_eigenvectors = fr_pca_results.components_
fr_eigenvalues = fr_pca_results.explained_variance_

print("Eigenvectors of Forward Rate Covariance Matrix:")
print(np.round(fr_eigenvectors, 3))
print("")
print("Eigenvalues of Forward Rate Covariance Matrix:")
print(fr_eigenvalues)
print("")


