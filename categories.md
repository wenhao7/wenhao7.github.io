---
layout: default
title: Categories
permalink: /categories/
exclude: false
---
<h2>List of categories in this blog</h2>
<ul>
{% assign categories = site.categories | sort: "title" %}
{% for post in categories %}
    <li> <a class="category-name" href="{{ post.url }}">{{ post.title }}</a>
    </li>
{% endfor %}
</ul>