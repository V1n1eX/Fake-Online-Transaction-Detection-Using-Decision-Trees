document.addEventListener('DOMContentLoaded', () => {
    const navLinks = Array.from(document.querySelectorAll('[data-nav]'));
    const sections = Array.from(document.querySelectorAll('[data-dashboard-section]'));

    const markActive = (sectionId) => {
        navLinks.forEach((link) => {
            const active = link.dataset.nav === sectionId;
            link.classList.toggle('is-active', active);
            if (active) {
                link.setAttribute('aria-current', 'location');
            } else {
                link.removeAttribute('aria-current');
            }
        });
    };

    navLinks.forEach((link) => {
        link.addEventListener('click', () => markActive(link.dataset.nav));
    });

    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            const visible = entries.filter((entry) => entry.isIntersecting)
                .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
            if (visible.length) markActive(visible[0].target.id);
        }, { rootMargin: '-12% 0px -72% 0px', threshold: 0 });
        sections.forEach((section) => observer.observe(section));
    }

    const form = document.getElementById('transaction-form');
    if (form) {
        form.addEventListener('submit', () => {
            const button = form.querySelector('.analyze-button');
            if (!button || button.disabled) return;
            button.disabled = true;
            button.setAttribute('aria-busy', 'true');
            button.querySelector('.button-label').hidden = true;
            button.querySelector('.button-loading').hidden = false;
            button.classList.add('is-loading');
        });

        window.addEventListener('pageshow', () => {
            const button = form.querySelector('.analyze-button');
            if (!button) return;
            button.disabled = false;
            button.removeAttribute('aria-busy');
            button.querySelector('.button-label').hidden = false;
            button.querySelector('.button-loading').hidden = true;
            button.classList.remove('is-loading');
        });
    }
});
