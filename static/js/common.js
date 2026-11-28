document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    lucide.createIcons();
  }

  const shell = document.getElementById("appShell");
  const sidebarToggle = document.getElementById("sidebarToggle");
  const mobileMenuBtn = document.getElementById("mobileMenuBtn");
  const notifBtn = document.getElementById("notifBtn");
  const notifPopover = document.getElementById("notifPopover");
  const globalDataset = document.getElementById("globalDatasetSelect");

  // Restore sidebar state
  if (localStorage.getItem("rgca_sidebar_collapsed") === "true") {
    shell?.classList.add("sidebar-collapsed");
  }

  sidebarToggle?.addEventListener("click", () => {
    shell?.classList.toggle("sidebar-collapsed");
    localStorage.setItem(
      "rgca_sidebar_collapsed",
      shell?.classList.contains("sidebar-collapsed") ? "true" : "false"
    );
  });

  mobileMenuBtn?.addEventListener("click", () => {
    shell?.classList.toggle("mobile-open");
  });

  notifBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    notifPopover?.classList.toggle("hidden");
  });

  document.addEventListener("click", (e) => {
    if (notifPopover && !notifPopover.contains(e.target) && e.target !== notifBtn) {
      notifPopover.classList.add("hidden");
    }
  });

  // Sync global dataset selector across pages
  if (globalDataset) {
    const savedDs = localStorage.getItem("rgca_dataset");
    if (savedDs && [...globalDataset.options].some((o) => o.value === savedDs)) {
      globalDataset.value = savedDs;
    }
    globalDataset.addEventListener("change", () => {
      localStorage.setItem("rgca_dataset", globalDataset.value);
      window.dispatchEvent(new CustomEvent("rgca:dataset-changed", { detail: globalDataset.value }));
    });
  }
});

window.RGCA = {
  formatPct(val) {
    if (val === null || val === undefined) return "—";
    return `${(val * 100).toFixed(1)}%`;
  },
  formatNum(val, digits = 3) {
    if (val === null || val === undefined) return "—";
    return Number(val).toFixed(digits);
  },
  labelBadge(label) {
    const map = {
      "Applies": "badge-success",
      "Not yet in force": "badge-warning",
      "Out of scope": "badge-neutral",
      "Wrong jurisdiction": "badge-neutral",
      "Heuristic only": "badge-blue",
      "Needs human review": "badge-danger",
    };
    const cls = map[label] || "badge-neutral";
    return `<span class="badge ${cls}">${label}</span>`;
  },
  refreshIcons() {
    if (window.lucide) lucide.createIcons();
  },
};
