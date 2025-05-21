// dashboard.js - Dashboard functionality

document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.querySelector('.main-content');
    const sidebarToggle = document.getElementById('sidebarToggle');
    
    // Check if sidebar state is stored in localStorage
    const sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
    
    // Set initial state based on localStorage or default to expanded on desktop
    if (sidebarCollapsed) {
        sidebar.classList.add('collapsed');
        mainContent.classList.add('expanded');
    }
    
    // Toggle sidebar on button click
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('collapsed');
            mainContent.classList.toggle('expanded');
            
            // Store sidebar state in localStorage
            localStorage.setItem('sidebarCollapsed', sidebar.classList.contains('collapsed'));
        });
    }
    
    // Expand sidebar on hover for desktop
    if (window.innerWidth > 992) {
        sidebar.addEventListener('mouseenter', function() {
            if (sidebar.classList.contains('collapsed')) {
                sidebar.classList.add('expanded');
            }
        });
        
        sidebar.addEventListener('mouseleave', function() {
            sidebar.classList.remove('expanded');
        });
    }
    
    // Highlight active menu item
    const currentPath = window.location.pathname;
    const menuItems = document.querySelectorAll('.menu-item');
    
    menuItems.forEach(item => {
        const link = item.querySelector('a');
        if (link && link.getAttribute('href') === currentPath) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
});
