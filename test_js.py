import streamlit as st
import streamlit.components.v1 as components

st.title("Test JS execution")
components.html("""
<script>
try {
    const parent = window.parent.document;
    parent.body.style.backgroundColor = "red";
    console.log("Success accessing parent!");
} catch(e) {
    console.error("Error accessing parent: " + e);
}
</script>
""", height=0)
