(function () {
    'use strict';

    var DEFAULT_BRAND = {
        productName: 'Nanobot',
        shortName: 'Nanobot',
        tagline: 'Invite-only AI companion',
        loginSubtitle: 'Enter invite code to start chatting',
        loginButtonLabel: 'Login',
        inviteLabel: 'Invite code',
        invitePlaceholder: 'Invite code',
        statusLabel: 'Online',
        sidebarTitle: 'Recent Chats',
        logoutLabel: 'Logout',
        newChatLabel: 'New Chat',
        messagePlaceholder: 'Type a message...',
        assistantName: 'Nanobot',
        assistantAvatarUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuAnTgNMSWaolorX1KbBnPvmYBhCltmdngCLe1-_mc3ZOtO6me-1HJfZsDr6MFEcrtCvHifvaHr6lEDGiRfmVfJ2rKecaU8sSFPrbJorycVKulM7iR4TqaSlxfVfq9dQxji_Gbx82L-b5W7SIVMnLhVIil_VZTQmQdg8TV1YvKGfRsD8hF-6Qn7TY6355PpBUka3JP0_M9ppdmVOvha_3SAUofzcs1gS3o147DcMrreGHN9c2vdYL6bMT1g1V7HPHO7_JDwO-yEmTgU',
        sidebarAvatarUrl: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBmAyeA5vP582WigANy5HsLC90At5fmigmBxhBB1gg7iwr03axrilFBradqnvkoZMIKCIseMwps_TrIW6OMl7Ho9jJFbUL5SJAf2s27alo15wV1KN10gh3UXq6NuS5ZZW7iFxvZZpJTmao_dL8kLUyQSQLIRG6tdCKCWBkyfi9ErYQmkvJT5ttsis3MCfbtrALXjvpywtYu1Nk0t-oFeEqh7HcXIRCXBoPlCuDXlEkMvF3oI83tBVql13iDC3GMf3afP5sUOg9sxww',
        userName: 'You',
        userAvatarIcon: 'person',
        userAvatarUrl: '',
        loginIcon: 'pets',
        loginIconUrl: '',
        themeColor: '#f59e0b',
        backgroundColor: '#f0fdf4',
        faviconUrl: '/icons/icon.svg',
        appleTouchIconUrl: '/icons/icon-192.png',
        manifestUrl: '/manifest.json'
    };

    function isNonEmptyString(value) {
        return typeof value === 'string' && value.trim().length > 0;
    }

    function applyText(key, value) {
        if (!isNonEmptyString(value)) return;
        var nodes = document.querySelectorAll('[data-brand-text="' + key + '"]');
        nodes.forEach(function (node) {
            node.textContent = value;
        });
    }

    function applyIcon(key, value) {
        if (!isNonEmptyString(value)) return;
        var nodes = document.querySelectorAll('[data-brand-icon="' + key + '"]');
        nodes.forEach(function (node) {
            node.textContent = value;
        });
    }

    function applyImage(key, value) {
        var nodes = document.querySelectorAll('[data-brand-img="' + key + '"]');
        nodes.forEach(function (node) {
            if (isNonEmptyString(value)) {
                node.setAttribute('src', value);
                node.classList.remove('hidden');
                if (key === 'loginIconUrl') {
                    var wrapper = node.closest('.login-icon');
                    if (wrapper) {
                        wrapper.classList.add('has-image');
                    }
                }
            } else {
                node.classList.add('hidden');
                if (key === 'loginIconUrl') {
                    var parent = node.closest('.login-icon');
                    if (parent) {
                        parent.classList.remove('has-image');
                    }
                }
            }
        });
    }

    function applyPlaceholder(key, value) {
        if (!isNonEmptyString(value)) return;
        var nodes = document.querySelectorAll('[data-brand-placeholder="' + key + '"]');
        nodes.forEach(function (node) {
            node.setAttribute('placeholder', value);
        });
    }

    function applyLink(rel, href) {
        if (!isNonEmptyString(href)) return;
        var link = document.querySelector('link[rel="' + rel + '"]');
        if (link) {
            link.setAttribute('href', href);
        }
    }

    function applyMeta(name, content) {
        if (!isNonEmptyString(content)) return;
        var meta = document.querySelector('meta[name="' + name + '"]');
        if (meta) {
            meta.setAttribute('content', content);
        }
    }

    function applyBrand(brand) {
        window.NANOBOT_BRAND = brand;

        if (isNonEmptyString(brand.productName)) {
            document.title = brand.productName;
        }

        applyMeta('apple-mobile-web-app-title', brand.productName);
        applyMeta('theme-color', brand.themeColor);
        applyLink('icon', brand.faviconUrl);
        applyLink('apple-touch-icon', brand.appleTouchIconUrl);
        applyLink('manifest', brand.manifestUrl);

        applyText('productName', brand.productName);
        applyText('tagline', brand.tagline);
        applyText('loginSubtitle', brand.loginSubtitle);
        applyText('loginButtonLabel', brand.loginButtonLabel);
        applyText('inviteLabel', brand.inviteLabel);
        applyText('statusLabel', brand.statusLabel);
        applyText('sidebarTitle', brand.sidebarTitle);
        applyText('logoutLabel', brand.logoutLabel);
        applyText('newChatLabel', brand.newChatLabel);
        applyText('assistantName', brand.assistantName);
        applyIcon('loginIcon', brand.loginIcon);
        applyImage('loginIconUrl', brand.loginIconUrl);
        applyPlaceholder('invitePlaceholder', brand.invitePlaceholder);
        applyPlaceholder('messagePlaceholder', brand.messagePlaceholder);

        var iconNode = document.querySelector('[data-brand-icon="loginIcon"]');
        if (iconNode) {
            if (isNonEmptyString(brand.loginIconUrl)) {
                iconNode.classList.add('hidden');
            } else {
                iconNode.classList.remove('hidden');
            }
        }

        if (isNonEmptyString(brand.sidebarAvatarUrl)) {
            document.documentElement.style.setProperty(
                '--brand-sidebar-avatar-url',
                'url("' + brand.sidebarAvatarUrl + '")'
            );
        }
        if (isNonEmptyString(brand.assistantAvatarUrl)) {
            document.documentElement.style.setProperty(
                '--brand-assistant-avatar-url',
                'url("' + brand.assistantAvatarUrl + '")'
            );
        }
    }

    function normalizeBrand(data) {
        if (!data || typeof data !== 'object') {
            return Object.assign({}, DEFAULT_BRAND);
        }
        return Object.assign({}, DEFAULT_BRAND, data);
    }

    function loadBrand() {
        return fetch('/brand.json', { cache: 'no-store' })
            .then(function (res) {
                if (!res.ok) {
                    throw new Error('brand.json not found');
                }
                return res.json();
            })
            .catch(function () {
                return {};
            });
    }

    applyBrand(DEFAULT_BRAND);

    window.NANOBOT_BRAND_READY = loadBrand()
        .then(function (data) {
            var merged = normalizeBrand(data);
            applyBrand(merged);
            return merged;
        })
        .catch(function () {
            return DEFAULT_BRAND;
        });
})();
