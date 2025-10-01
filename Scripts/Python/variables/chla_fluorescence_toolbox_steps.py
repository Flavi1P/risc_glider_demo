#from toolbox.steps.base_step import BaseStep, register_step
#import toolbox.utils.diagnostics as diag

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


@register_step
class QCChlorophyll(BaseStep):
    step_name = "QC: Chlorophyll"


@register_step
class AdjustChlorophyll(BaseStep):
    step_name = "ADJ: Chlorophyll"

    def run(self):
        """
        Apply dark value correction for Chlorophyll-a.
        
        The dark value represents the sensor's baseline reading in the absence of
        chlorophyll fluorescence. This correction removes instrumental offset by:
        1. Checking config file for user-specified dark value
        2. If not present, computing dark value from deep profiles (>200m)
        3. Applying correction: CHLA_corrected = CHLA_raw - dark_value
        
        Parameters
        ----------
        self.context["data"] : xarray.Dataset
            Raw glider dataset, which should contain:
            - TIME: time [numpy.datetime64]
            - DEPTH or PRES: depth [m] or pressure [dbar]
            - CHLA: raw chlorophyll-a fluorescence [mg/m³]
            - PROFILE_NUMBER: profile index
            
        Returns
        -------
        self.context : dict
            Updated context with corrected CHLA_ADJ variable
        """
        self.log(f"Running CHLA ADJ")
        self.log(
            f"Run with diagnostics to determine dark value and visualize correction. Diagnostics: {self.diagnostics}"
        )

        # Check if the data is in the context
        if "data" not in self.context:
            raise ValueError("No data found in context. Please load data first.")
        
        self.data = self.context["data"]
        
        # Get or compute dark value
        self.dark_value = self.get_dark_value()
        self.log(f"Using dark value: {self.dark_value:.6f}")
        
        # Apply dark value correction
        self.apply_dark_correction()
        
        # Generate diagnostics if requested
        if self.diagnostics:
            self.generate_diagnostics()
        
        self.context["data"] = self.data
        return self.context

    def get_dark_value(self):
        """
        Get dark value from config or compute from deep profiles.
        
        The dark value is computed as the median of minimum CHLA values from
        the first 5 profiles that reach at least 200m depth.
        
        Returns
        -------
        float
            Dark value for chlorophyll-a correction
        """
        # Check if user has specified dark value in config
        if "dark_value" in self.parameters and self.parameters["dark_value"] is not None:
            self.log("Using user-specified dark value from config")
            return self.parameters["dark_value"]
        
        self.log("Computing dark value from deep profiles (>200m)")
        
        # Find profiles reaching at least 200m
        prof_idx = self.data.PROFILE_NUMBER.values
        unique_prof = np.unique(prof_idx[~np.isnan(prof_idx)])
        
        # Get depth variable (could be DEPTH or computed from PRES)
        if "DEPTH" in self.data.variables:
            depth = self.data.DEPTH.values
        elif "PRES" in self.data.variables:
            depth = self.data.PRES.values  # Using pressure as proxy for depth
        else:
            raise ValueError("No DEPTH or PRES variable found in dataset")
        
        # Find first 5 profiles reaching >= 200m
        deep_profiles = []
        self.dark_profiles = []  # Store for diagnostics
        
        for prof in unique_prof:
            if len(deep_profiles) >= 5:
                break
            
            prof_mask = prof_idx == prof
            prof_depth = depth[prof_mask]
            
            # Check if profile reaches 200m
            if np.nanmax(prof_depth) >= 200:
                deep_profiles.append(prof)
                self.dark_profiles.append(prof)
        
        if len(deep_profiles) == 0:
            self.log("WARNING: No profiles reaching 200m found. Using all available data.")
            # Fallback: use deepest available data from first 5 profiles
            deep_profiles = unique_prof[:5]
            self.dark_profiles = deep_profiles.tolist()
        
        self.log(f"Found {len(deep_profiles)} profiles for dark value calculation: {deep_profiles}")
        
        # Compute minimum CHLA for each deep profile (from 200m and below)
        min_values = []
        self.dark_profile_data = {}  # Store for diagnostics
        
        for prof in deep_profiles:
            prof_mask = prof_idx == prof
            prof_depth = depth[prof_mask]
            prof_chla = self.data.CHLA.values[prof_mask]
            
            # Get data from 200m and deeper
            deep_mask = prof_depth >= 200
            if np.sum(deep_mask) > 0:
                deep_chla = prof_chla[deep_mask]
                min_val = np.nanmin(deep_chla)
            else:
                # Fallback: use deepest 10% of profile
                n_deep = max(int(0.1 * len(prof_chla)), 1)
                sorted_indices = np.argsort(prof_depth)[-n_deep:]
                min_val = np.nanmin(prof_chla[sorted_indices])
            
            if np.isfinite(min_val):
                min_values.append(min_val)
                # Store profile data for diagnostics
                self.dark_profile_data[prof] = {
                    'depth': prof_depth,
                    'chla': prof_chla,
                    'min_value': min_val
                }
        
        if len(min_values) == 0:
            raise ValueError("Could not compute dark value: no valid CHLA data in deep profiles")
        
        # Compute median of minimum values
        dark_value = np.nanmedian(min_values)
        
        # Save dark value to config/parameters
        self.parameters["dark_value"] = float(dark_value)
        self.log(f"Computed dark value: {dark_value:.6f} (from {len(min_values)} profiles)")
        
        # Save to config file if available
        if hasattr(self, 'config_manager'):
            try:
                self.config_manager.set('chlorophyll', 'dark_value', dark_value)
                self.log("Saved dark value to config file")
            except Exception as e:
                self.log(f"Warning: Could not save dark value to config: {e}")
        
        return dark_value

    def apply_dark_correction(self):
        """
        Apply dark value correction to CHLA data.
        
        Creates CHLA_ADJ variable: CHLA_ADJ = CHLA - dark_value
        """
        # Create adjusted chlorophyll variable
        self.data["CHLA_ADJ"] = xr.DataArray(
            self.data.CHLA.values - self.dark_value,
            dims=self.data.CHLA.dims,
            coords=self.data.CHLA.coords,
        )
        
        # Copy and update attributes
        self.data["CHLA_ADJ"].attrs = self.data.CHLA.attrs.copy()
        self.data["CHLA_ADJ"].attrs["comment"] = (
            f"CHLA with dark value correction (dark_value={self.dark_value:.6f})"
        )
        self.data["CHLA_ADJ"].attrs["dark_value"] = self.dark_value
        
        self.log("Applied dark value correction to CHLA")

    def generate_diagnostics(self):
        """
        Generate diagnostic plots for dark value correction.
        """
        plt.ion()
        self.log("Generating diagnostics...")
        
        self.display_dark_profiles()
        self.display_chla_transects()
        
        plt.ioff()
        self.log("Diagnostics generated.")

    def display_dark_profiles(self):
        """
        Display the 5 profiles used to compute the dark value.
        Shows depth vs CHLA with minimum values highlighted.
        """
        if not hasattr(self, 'dark_profile_data'):
            self.log("No dark profile data available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.dark_profile_data)))
        
        for idx, (prof, data) in enumerate(self.dark_profile_data.items()):
            depth = data['depth']
            chla = data['chla']
            min_val = data['min_value']
            
            # Plot profile
            ax.plot(
                chla,
                -depth,
                color=colors[idx],
                linewidth=2,
                label=f"Profile {int(prof)}: min={min_val:.4f}",
                alpha=0.7
            )
            
            # Highlight minimum value
            min_idx = np.nanargmin(chla[depth >= 200]) if np.any(depth >= 200) else np.nanargmin(chla)
            min_depth = depth[depth >= 200][min_idx] if np.any(depth >= 200) else depth[np.nanargmin(chla)]
            ax.plot(
                min_val,
                -min_depth,
                'o',
                color=colors[idx],
                markersize=10,
                markeredgecolor='black',
                markeredgewidth=1.5
            )
        
        # Add dark value line
        ax.axvline(
            self.dark_value,
            color='red',
            linestyle='--',
            linewidth=2.5,
            label=f'Dark Value: {self.dark_value:.4f}'
        )
        
        # Add 200m reference line
        ax.axhline(-200, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.95, -200, '200m', 
                verticalalignment='bottom', horizontalalignment='right',
                fontsize=10, color='gray')
        
        ax.set_xlabel('Chlorophyll-a [mg/m³]', fontweight='bold', fontsize=12)
        ax.set_ylabel('Depth [m]', fontweight='bold', fontsize=12)
        ax.set_title('Dark Value Calculation - Deep Profiles', 
                     fontweight='bold', fontsize=14)
        ax.legend(loc='best', prop={'size': 10})
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()

    def display_chla_transects(self):
        """
        Display before and after transects of CHLA.
        Shows raw CHLA and corrected CHLA_ADJ side by side.
        """
        # Select subset of data for clearer visualization
        prof_idx = self.data.PROFILE_NUMBER.values
        unique_prof = np.unique(prof_idx[~np.isnan(prof_idx)])
        
        # Use middle 40 profiles or all if fewer
        n_prof = len(unique_prof)
        if n_prof > 40:
            prof_start = unique_prof[int(n_prof * 0.3)]
            prof_end = unique_prof[int(n_prof * 0.7)]
            subset = self.data.where(
                (prof_idx >= prof_start) & (prof_idx <= prof_end), 
                drop=True
            )
        else:
            subset = self.data
        
        # Get depth variable
        if "DEPTH" in subset.variables:
            depth = -subset.DEPTH.values
        elif "PRES" in subset.variables:
            depth = -subset.PRES.values
        else:
            depth = np.zeros_like(subset.TIME.values)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 10), sharex=True)
        
        # Plot raw CHLA
        chla_raw = subset.CHLA.values
        vmin_raw = np.nanpercentile(chla_raw, 5)
        vmax_raw = np.nanpercentile(chla_raw, 95)
        
        sc1 = ax1.scatter(
            subset.TIME,
            depth,
            c=chla_raw,
            cmap='YlGnBu',
            vmin=vmin_raw,
            vmax=vmax_raw,
            s=10
        )
        cbar1 = plt.colorbar(sc1, ax=ax1)
        cbar1.set_label('CHLA [mg/m³]', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Depth [m]', fontweight='bold', fontsize=12)
        ax1.set_title('Raw Chlorophyll-a', fontweight='bold', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Plot corrected CHLA
        chla_adj = subset.CHLA_ADJ.values
        vmin_adj = np.nanpercentile(chla_adj, 5)
        vmax_adj = np.nanpercentile(chla_adj, 95)
        
        sc2 = ax2.scatter(
            subset.TIME,
            depth,
            c=chla_adj,
            cmap='YlGnBu',
            vmin=vmin_adj,
            vmax=vmax_adj,
            s=10
        )
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_label('CHLA_ADJ [mg/m³]', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Depth [m]', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Time', fontweight='bold', fontsize=12)
        ax2.set_title(
            f'Corrected Chlorophyll-a (Dark Value: {self.dark_value:.4f})', 
            fontweight='bold', 
            fontsize=14
        )
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Optional: Create difference plot
        fig2, ax3 = plt.subplots(figsize=(14, 6))
        
        diff = chla_raw - chla_adj
        vmax_diff = np.nanpercentile(np.abs(diff), 95)
        
        sc3 = ax3.scatter(
            subset.TIME,
            depth,
            c=diff,
            cmap='RdBu_r',
            vmin=-vmax_diff,
            vmax=vmax_diff,
            s=10
        )
        cbar3 = plt.colorbar(sc3, ax=ax3)
        cbar3.set_label('CHLA - CHLA_ADJ [mg/m³]', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Depth [m]', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Time', fontweight='bold', fontsize=12)
        ax3.set_title(
            'Dark Value Correction Difference', 
            fontweight='bold', 
            fontsize=14
        )
        ax3.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()