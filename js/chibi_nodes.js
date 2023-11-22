import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
  name: "chibi_nodes",

  beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "Textbox") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        return r;
      };

      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);

        for (const widget of this.widgets) {
          if (widget.type === "customtext") {
            widget.value = message.text.join("");
          }
        }

        this.onResize?.(this.size);
      };
    }
  },
});
